import argparse
import contextlib
import os
import random
import sys

import numpy as np
from matplotlib import pyplot as plt
from torch.backends import mps
from tqdm import tqdm

import config
from bigram_transformer import *

dataset.generate_batches()

batch_size = 32
learning_rate = 0.00025
val_interval = 5
gradient_acc = 4
epochs = int(sys.argv[1])
use_mixed_precision = False

val_loss_history = []
training_loss_history = []

device = "cuda" if torch.cuda.is_available() else "cpu"
if mps.is_built():
    device = torch.device("mps")

# train and test splits
data = dataset.prep_data
n = int(0.97 * len(data))
train_data = data[:n]
val_data = data[n:]


# model with hyperparams
model = BigramLanguageModel(
    block_size=config.BLOCK_SIZE,
    n_embd=config.N_EMBD,
    n_layers=config.N_LAYERS,
    n_head=config.N_HEAD,
    dropout=0.2,
).to_device(device)
model.train()


def main():
    if os.path.exists(model.transformer_model_name):
        model.load(True, map_location=device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) // 1_000_000, "M parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=10)

    t = tqdm(range(epochs))
    val_loss = 0
    total_loss = 0
    for iter in t:
        xb, yb = get_batch("train")

        with (
            torch.autocast(model.device)
            if use_mixed_precision and model.device == "cuda"
            else contextlib.nullcontext()
        ):
            if (iter + 1) % val_interval == 0:
                val_loss = get_val_loss(model, 1)

            # evaluate the loss
            _, loss = model(xb, yb)
            val_loss_history.append(val_loss)
            training_loss_history.append(loss.item())
            loss: torch.Tensor = loss / gradient_acc

        loss.backward()
        nn.utils.clip_grad.clip_grad_norm(model.parameters(), 1e-3)
        total_loss += loss.item()
        scheduler.step()

        if (iter + 1) % gradient_acc == 0 or (iter + 1) == epochs:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            t.set_description(
                f"Epoch {iter} - Train loss: {total_loss:.4f}  Validation loss: {round(val_loss, 5) if val_loss else 'N/A'} LR: {scheduler.get_lr()[-1]}"
            )
            total_loss = 0

    model.save()
    show_loss()


def show_loss():
    """Display training and validation loss history on graph"""

    epoch_l = list(range(1, epochs + 1))
    plt.plot(epoch_l, training_loss_history, label="Training loss")
    plt.plot(epoch_l, val_loss_history, label="Validation loss")
    plt.legend()
    plt.show()


@torch.no_grad()
def get_val_loss(model: BigramLanguageModel, eval_iters=50) -> float:
    """Estimates the validation loss of current model"""

    model.eval()

    val_loss = 0.0
    for _ in range(eval_iters):
        X, Y = get_batch("val")

        _, loss = model(X, Y)
        val_loss += loss.item()

    # get the mean
    val_loss /= eval_iters
    model.train()

    return val_loss


def get_batch(split):
    """Get a randomly sampled batch of data"""

    ds = train_data if split == "train" else val_data
    batch = [ds[random.randint(0, len(ds) - 1)] for _ in range(batch_size)]

    x = []
    y = []
    for a, b in batch:
        x.append(a)
        y.append(b)

    return torch.from_numpy(np.array(x)).to(device), torch.from_numpy(np.array(y)).to(
        device
    )


class HyperparamArgs:
    """Process arguments from argparse into a model"""

    def __init__(self, namespace: argparse.Namespace):
        self.epochs: int = namespace.epochs
        self.batch_size: int = namespace.batch_size
        self.gradient_acc: int = namespace.gradient_acc
        self.use_mixed_precision: bool = namespace.use_mixed_precision
        self.dropout: float = namespace.dropout


def parse_arguments() -> HyperparamArgs:
    """Parse argument switches from command line"""

    parser = argparse.ArgumentParser()
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

    return HyperparamArgs(parser.parse_args())


if __name__ == "__main__":
    main()
