import argparse
import contextlib
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.backends import mps
from tqdm import tqdm

from src import logger
from src.config import Config

from .transformer import *

dataset.generate_batches()

# train and test splits
data = dataset.prep_data
n = int(0.97 * len(data))
train_data = data[:n]
val_data = data[n:]
val_interval = 5

val_loss_history = []
training_loss_history = []

device = "cuda" if torch.cuda.is_available() else "cpu"
if mps.is_built():
    device = torch.device("mps")


def main():
    args: HyperparamArgs = parse_arguments()
    logger.special(args)

    # model with hyperparams
    runner = TransientRunner(
        block_size=Config.BLOCK_SIZE,
        n_embd=Config.N_EMBD,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        dropout=args.dropout,
    )
    runner.to_device(device)

    if os.path.exists(runner.transformer_model_name):
        runner.load(True, map_location=device)
    
    runner.use_parallel()
    runner.model.train()

    # print the number of parameters in the model
    logger.info(sum(p.numel() for p in runner.model.parameters()) // 1_000_000, "M parameters")
    optimizer = torch.optim.AdamW(
        runner.model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=10)

    val_loss = 0
    total_loss = 0

    t = tqdm(range(args.epochs))
    if args.in_jupyter:
        from tqdm import tqdm_notebook
        t = tqdm_notebook(range(args.epochs))
        
    for iter in t:
        xb, yb = get_batch("train", args.batch_size)

        with (
            torch.autocast(runner.device)
            if args.use_mixed_precision and runner.device == "cuda"
            else contextlib.nullcontext()
        ):
            if (iter + 1) % val_interval == 0:
                val_loss = get_val_loss(runner.model, args.batch_size, eval_iters=1)

            # evaluate the loss
            _, loss = runner.forward(xb, yb)
            val_loss_history.append(val_loss)
            training_loss_history.append(loss.mean().item())
            loss: torch.Tensor = loss / args.gradient_acc

        loss.mean().backward()
        nn.utils.clip_grad.clip_grad_norm(runner.model.parameters(), 1e-3)
        total_loss += loss.mean().item()
        scheduler.step()

        if (iter + 1) % args.gradient_acc == 0 or (iter + 1) == args.epochs:
            optimizer.step()
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
def get_val_loss(model: TransformerModel, batch_size, eval_iters=50) -> float:
    """Estimates the validation loss of current model"""

    model.eval()

    val_loss = 0.0
    for _ in range(eval_iters):
        X, Y = get_batch("val", batch_size)

        _, loss = model(X, Y, device)
        val_loss += loss.mean().item()

    # get the mean
    val_loss /= eval_iters
    model.train()

    return val_loss


def get_batch(split, batch_size):
    """Get a randomly sampled batch of data"""

    ds = train_data if split == "train" else val_data
    batch = [ds[random.randint(0, len(ds) - 1)] for _ in range(batch_size)]

    x = []
    y = []
    for a, b in batch:
        x.append(a)
        y.append(b)

    return (
        torch.from_numpy(
            np.array(x)
        ).to(device),
        torch.from_numpy(
            np.array(y)
        ).to(device)
    )


class HyperparamArgs:
    """Process arguments from argparse into a model"""

    def __init__(self, namespace: argparse.Namespace):
        self.lr: float = namespace.lr
        self.epochs: int = namespace.epochs
        self.batch_size: int = namespace.batch_size
        self.gradient_acc: int = namespace.gradient_acc
        self.use_mixed_precision: bool = namespace.use_mixed_precision
        self.dropout: float = namespace.dropout
        self.in_jupyter: bool = namespace.in_jupyter

    def __repr__(self):
        return f"""Hyperparams:
        lr: {self.lr}
        epochs: {self.epochs}
        batch_size: {self.batch_size}
        gradient_acc: {self.gradient_acc}
        use_mixed_precision: {self.use_mixed_precision}
        dropout: {self.dropout}
        in_jupyter: {self.in_jupyter}
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
        default=True,
        type=bool,
        help="Use automatic precision to speed up training (only on CUDA-enabled GPUs)",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default=0.2,
        type=float,
        help="Dropout rate to drop out weights to reduce overfitting",
    )
    parser.add_argument(
        "-j",
        "--in-jupyter",
        default=False,
        type=bool,
        help="Set to true if running in Jupyter Notebook",
    )

    return HyperparamArgs(parser.parse_args())


if __name__ == "__main__":
    main()
