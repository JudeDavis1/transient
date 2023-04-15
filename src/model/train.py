import argparse
import contextlib
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.backends import mps
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
from torch.cuda.amp import GradScaler, autocast
from src import logger
from src.config import Config

from .transformer import *

dataset.generate_batches()

# train and test splits
data = dataset.prep_data
n = int(0.97 * len(data))
val_interval = 5

val_loss_history = []
training_loss_history = []

device = "cuda" if torch.cuda.is_available() else "cpu"
if mps.is_built():
    device = torch.device("mps")


def main():
    args: HyperparamArgs = parse_arguments()
    logger.special(args)

    train_data = DataLoader(data[:n], batch_size=args.batch_size, shuffle=True)
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

    if os.path.exists(args.from_pretrained):
        runner.load(args.from_pretrained, map_location=device)
    
    runner.use_parallel_if_available()
    runner.model.train()

    # print the number of parameters in the model
    logger.info(sum(p.numel() for p in runner.model.parameters()) // 1_000_000, "M parameters")
    optimizer = torch.optim.AdamW(
        runner.model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=15)
    scaler = GradScaler() if args.use_mixed_precision and device == 'cuda' else FakeGradScaler()

    val_loss = 0
    total_loss = 0

    if args.in_jupyter:
        t = tqdm_notebook(range(args.epochs))
    else:
        t = tqdm(range(args.epochs))
        
    for iter in t:
        xb, yb = get_batch(train_data)

        with (
            autocast(enabled=args.use_mixed_precision and device == 'cuda')
        ):
            if (iter + 1) % val_interval == 0:
                val_loss = get_val_loss(runner.model, val_data, eval_iters=1)

            # evaluate the loss
            _, loss = runner.forward(xb, yb)
            val_loss_history.append(val_loss)
            training_loss_history.append(loss.mean().item())
            loss: torch.Tensor = loss / args.gradient_acc
        
        scaler.scale(loss.mean()).backward()
        nn.utils.clip_grad.clip_grad_norm_(runner.model.parameters(), max_norm=3.0)

        total_loss += loss.mean().item()
        scheduler.step()

        if (iter + 1) % args.gradient_acc == 0 or (iter + 1) == args.epochs:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t.set_description(
                f"Epoch {iter} - Train loss: {total_loss:.4f}  Validation loss: {round(val_loss, 5) if val_loss else 'N/A'} LR: {scheduler.get_lr()[-1]}"
            )
            total_loss = 0

    runner.save()
    show_loss(args.epochs)


def show_loss(epochs):
    """Display training and validation loss history on graph"""

    epoch_l = list(range(1, epochs + 1))
    plt.plot(epoch_l, training_loss_history, label="Training loss")
    plt.plot(epoch_l, val_loss_history, label="Validation loss")
    plt.legend()
    plt.show()


@torch.no_grad()
def get_val_loss(model: TransformerModel, dataloader, eval_iters=50) -> float:
    """Estimates the validation loss of current model"""

    model.eval()

    val_loss = 0.0
    for _ in range(eval_iters):
        X, Y = get_batch(dataloader)

        _, loss = model(X, Y, device)
        val_loss += loss.mean().item()

    # get the mean
    val_loss /= eval_iters
    model.train()

    return val_loss


def get_batch(dataloader):
    """Get a randomly sampled batch of data"""

    x, y = next(iter(dataloader))

    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True)
    )

class FakeGradScaler():
    """A placeholder GradScaler for when mixed precision is not available"""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self): return


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
        default=0.,
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

    return HyperparamArgs(parser.parse_args())


if __name__ == "__main__":
    main()
