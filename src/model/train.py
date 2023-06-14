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

dataset.generate_batches(Config.BLOCK_SIZE)

# train and test splits
data = dataset.prep_data
n = int(0.98 * len(data))
val_interval = 8
optimizer_checkpoint_name = "optimizer_cache"
min_lr = 0.00004
grad_max_norm = 4.0
max_warmup_steps = 1000
should_warmup = True

val_loss_history = []
training_loss_history = []

# # for unsupported GPUs when compiling the model
# torch._C._dynamo.config.suppress_errors = True
device = "cuda" if torch.cuda.is_available() else "cpu"
if mps.is_built():
    device = torch.device("mps")


def main():
    global device, min_lr, should_warmup
    args: HyperparamArgs = parse_arguments()
    logger.special(args)

    if args.dropout:
        min_lr = 0.00002

    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    
    random.shuffle(data)
    train_data = DataLoader(data[:n], batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_data = DataLoader(data[n:], batch_size=args.batch_size, shuffle=True)

    # model with hyperparams
    runner = TransientRunner(
        block_size=Config.BLOCK_SIZE,
        n_embd=Config.N_EMBD,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        dropout=args.dropout,
    )
    runner.to_device(device)

    # try:
    #     runner.compile_model()
    # except RuntimeError as e:
    #     print(e)

    pretrained_model_exists = os.path.exists(args.from_pretrained)
    if pretrained_model_exists:
        runner.load(args.from_pretrained, map_location=device)

    runner.use_parallel_if_available()
    runner.model.train()

    # print the number of parameters in the model
    logger.info(
        sum(p.numel() for p in runner.model.parameters()) // 1_000_000, "M parameters"
    )
    optimizer = torch.optim.AdamW(
        runner.model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1
    )
    if os.path.exists(optimizer_checkpoint_name) and pretrained_model_exists:
        should_warmup = False
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_name))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=20)
    scaler = (
        GradScaler()
        if args.use_mixed_precision and device == "cuda"
        else FakeGradScaler()
    )

    val_loss = 0
    total_loss = 0

    if args.in_jupyter:
        t = tqdm_notebook(range(args.epochs))
    else:
        t = tqdm(range(args.epochs))

    n_steps_per_batch = len(train_data)
    n_steps = args.epochs * n_steps_per_batch

    # set_lr(optimizer, 0.00006)

    for iter in t:
        for j, (xb, yb) in enumerate(train_data):
            if scheduler.get_lr()[-1] < min_lr:
                set_lr(optimizer, min_lr)

            cur_step = (iter * len(train_data)) + j
            if should_warmup and cur_step <= max_warmup_steps:
                new_lr = (args.lr / max_warmup_steps) * cur_step
                set_lr(optimizer, new_lr)

            xb = xb.to(device)
            yb = yb.to(device)

            # with mixed precision
            with autocast(enabled=args.use_mixed_precision and device == "cuda"):
                if (cur_step + 1) % val_interval == 0:
                    val_loss = get_val_loss(runner.model, val_data, eval_iters=4)

                # evaluate the loss
                _, loss = runner.forward(xb, yb)
                val_loss_history.append(val_loss)
                training_loss_history.append(loss.mean().item())
                loss: torch.Tensor = loss / args.gradient_acc

                scaler.scale(loss.mean()).backward()
                nn.utils.clip_grad.clip_grad_norm_(runner.model.parameters(), max_norm=grad_max_norm)

                total_loss += loss.mean().item()
                scheduler.step()

                if (cur_step + 1) % args.gradient_acc == 0 or (cur_step + 1) == n_steps:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    val_loss_str = round(val_loss, 6) if val_loss else "N/A"
                    lr_str = scheduler.get_lr()[-1]
                    t.set_description(
                        f"Epoch {iter} - Batch: {j + 1}/{n_steps_per_batch} - Train loss: {total_loss:.6f}  Validation loss: {val_loss_str}  LR: {lr_str:.7f}"
                    )
                    total_loss = 0

    runner.save(args.save_to)
    torch.save(optimizer.state_dict(), optimizer_checkpoint_name)
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

        _, loss = model(X.to(device), Y.to(device), device)
        val_loss += loss.mean().item()

    # get the mean
    val_loss /= eval_iters
    model.train()

    return val_loss


def get_batch(dataloader: DataLoader):
    """Get a randomly sampled batch of data"""

    x, y = next(iter(dataloader))

    return (x.to(device), y.to(device))


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
        default=1,
        type=int,
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

    return HyperparamArgs(parser.parse_args())


if __name__ == "__main__":
    main()
