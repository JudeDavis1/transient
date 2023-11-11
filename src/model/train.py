import os
import torch
import random
import argparse

from matplotlib import pyplot as plt
from torch.backends import mps
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

from src import logger
from src.config import Config
from src.model.transformer import *


dataset.load_dataset()
dataset.generate_batches(Config.BLOCK_SIZE)

# train and test splits
DATA = dataset.batch_data
N = int(0.98 * len(DATA))
VALIDATION_INTERVAL = 3
OPTIMIZER_CHECKPOINT_NAME = "optimizer_cache"
MIN_LR = 0.00004
GRAD_MAX_NORM = 1.0
MAX_WARMUP_STEPS = 50
CHECKPOINT_INTERVAL = 15
SHOULD_WARMUP = True

val_loss_history = []
training_loss_history = []

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if mps.is_built():
    DEVICE = "mps"


def main():
    global DEVICE, MIN_LR, SHOULD_WARMUP
    args: HyperparamArgs = parse_arguments()
    logger.special(args)

    if args.dropout:
        MIN_LR = 0.00002

    if args.device.startswith("xla"):
        import torch_xla as xm
        DEVICE = xm.core.xla_model.xla_device()
    
    random.shuffle(DATA)
    train_data = DataLoader(DATA[:N], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_data = DataLoader(DATA[N:], batch_size=args.batch_size, shuffle=True)

    # model with hyperparams
    runner = TransientRunner(
        block_size=Config.BLOCK_SIZE,
        n_embd=Config.N_EMBD,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        dropout=args.dropout,
    )
    runner.to_device(DEVICE)
    
    if args.compile:
        runner.compile_model()

    pretrained_model_exists = os.path.exists(args.from_pretrained)
    if pretrained_model_exists:
        runner.load(args.from_pretrained, map_location=DEVICE)

    runner.use_parallel_if_available()
    runner.model.train()

    # print the number of parameters in the model
    logger.info(
        sum(p.numel() for p in runner.model.parameters() if p.requires_grad) // 1_000_000, "M parameters"
    )
    optimizer = torch.optim.AdamW(
        runner.model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1
    )
    if os.path.exists(OPTIMIZER_CHECKPOINT_NAME) and pretrained_model_exists:
        SHOULD_WARMUP = False
        optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_NAME))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=20)
    scaler = (
        GradScaler()
        if args.use_mixed_precision and DEVICE == "cuda"
        else FakeGradScaler()
    )

    val_loss = 0
    total_loss = 0

    if args.in_jupyter:
        t = tqdm_notebook(range(args.epochs))
    else:
        t = tqdm(range(args.epochs))

    n_steps_per_batch = len(train_data) // args.gradient_acc
    n_steps = args.epochs * (n_steps_per_batch)
    if args.in_jupyter:
        t = tqdm_notebook(range(args.epochs))
    else:
        t = tqdm(range(args.epochs))
    
    completed_updates = 0
    for iter in t:
        for j, (xb, yb) in enumerate(train_data):
            if scheduler.get_lr()[-1] < MIN_LR:
                set_lr(optimizer, MIN_LR)

            cur_step = (iter * len(train_data)) + j
            if SHOULD_WARMUP and cur_step <= MAX_WARMUP_STEPS:
                new_lr = (args.lr / MAX_WARMUP_STEPS) * cur_step
                set_lr(optimizer, new_lr)

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # with mixed precision
            with autocast(enabled=args.use_mixed_precision and DEVICE == "cuda"):
                if completed_updates % VALIDATION_INTERVAL == 0:
                    val_loss = get_val_loss(runner.model, val_data, eval_iters=2)

                # evaluate the loss
                _, loss = runner.forward(xb, yb)
                loss: torch.Tensor = loss / args.gradient_acc

            scaler.scale(loss.mean()).backward()
            nn.utils.clip_grad.clip_grad_norm_(runner.model.parameters(), max_norm=GRAD_MAX_NORM)

            total_loss += loss.mean().item()
            scheduler.step()

            if (cur_step + 1) % args.gradient_acc == 0 or (cur_step + 1) == n_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                val_loss_history.append(val_loss)
                training_loss_history.append(loss.mean().item())

                val_loss_str = round(val_loss, 6) if val_loss else "N/A"
                lr_str = scheduler.get_lr()[-1]

                update_num = (j // args.gradient_acc) + 1
                t.set_description(
                    f"Epoch {iter} - Batch: {update_num}/{n_steps_per_batch} - Train loss: {total_loss:.6f}  Validation loss: {val_loss_str}  LR: {lr_str:.7f}"
                )
                total_loss = 0
                completed_updates += 1
            
                if completed_updates % CHECKPOINT_INTERVAL == 0:
                    runner.save(args.save_to, verbose=True)

    runner.save(args.save_to, verbose=True)
    torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_NAME)
    show_loss(n_steps)


def set_lr(optimizer, new_lr):
    optimizer.param_groups[0]['lr'] = new_lr

def show_loss(epochs):
    """Display training and validation loss history on graph"""

    epoch_l = list(range(1, epochs + 1))
    plt.plot(epoch_l, training_loss_history, label="Training loss")
    plt.plot(epoch_l, val_loss_history, label="Validation loss")
    plt.legend()
    plt.savefig("loss_history.png")


@torch.no_grad()
def get_val_loss(model: TransformerModel, dataloader, eval_iters=50) -> float:
    """Estimates the validation loss of current model"""

    model.eval()

    val_loss = 0.0
    for _ in range(eval_iters):
        X, Y = get_batch(dataloader)

        _, loss = model(X.to(DEVICE), Y.to(DEVICE), DEVICE)
        val_loss += loss.mean().item()

    # get the mean
    val_loss /= eval_iters
    model.train()

    return val_loss


def get_batch(dataloader: DataLoader):
    """Get a randomly sampled batch of data"""

    x, y = next(iter(dataloader))

    return (x.to(DEVICE), y.to(DEVICE))


class FakeGradScaler:
    """A placeholder GradScaler for when mixed precision is not available"""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return


class HyperparamArgs:
    """Process arguments from argparse into a model"""

    def __init__(self, namespace: argparse.Namespace):
        self.lr: float = namespace.lr
        self.epochs: int = namespace.epochs
        self.batch_size: int = namespace.batch_size
        self.gradient_acc: int = namespace.gradient_acc
        self.use_mixed_precision: bool = bool(namespace.use_mixed_precision)
        self.dropout: float = namespace.dropout
        self.in_jupyter: bool = bool(namespace.in_jupyter)
        self.from_pretrained: str = namespace.from_pretrained
        self.device: str = namespace.device
        self.save_to: str = namespace.save_to
        self.compile: bool = namespace.compile

    def __repr__(self):
        return f"""Hyperparams:
        lr: {self.lr}
        epochs: {self.epochs}
        batch_size: {self.batch_size}
        gradient_acc: {self.gradient_acc}
        use_mixed_precision: {self.use_mixed_precision}
        dropout: {self.dropout}
        in_jupyter: {self.in_jupyter}
        from_pretrained: {self.from_pretrained}
        device: {self.device}
        save_to: {self.save_to}
        """


def parse_arguments() -> HyperparamArgs:
    """Parse argument switches from command line"""

    parser = argparse.ArgumentParser(
        description="Train the transformer base-model on unstructured text."
    )
    parser.add_argument(
        "--lr",
        default=0.0003,
        type=float,
        help="Set beginning learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (training iterations)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="Number of batches to process at once (per step)",
    )
    parser.add_argument(
        "-ga",
        "--gradient-acc",
        default=4,
        type=int,
        help="Number of gradient accumulation steps (simulate larger batch size by accumulating gradients per step)",
    )
    parser.add_argument(
        "-mp",
        "--use-mixed-precision",
        action="store_true",
        help="Use automatic precision to speed up training (only on CUDA-enabled GPUs)",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate to randomly drop out weights to reduce overfitting",
    )
    parser.add_argument(
        "-j",
        "--in-jupyter",
        default=0,
        type=int,
        help="Set to true if running in Jupyter Notebook",
    )
    parser.add_argument(
        "-f",
        "--from-pretrained",
        default="model_cache",
        type=str,
        help="Pretrained file checkpoint to load",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Accelerator device to use (supports: cuda, cpu, mps, xla.)",
    )
    parser.add_argument(
        "--save-to",
        default="model_cache",
        type=str,
        help="File to save to",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch dynamo",
    )

    return HyperparamArgs(parser.parse_args())


if __name__ == "__main__":
    main()
