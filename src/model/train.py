import os
import random

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

from src import logger
from src.config import config
from src.model.train_argparser import HyperparamArgs, parse_arguments
from src.model.transformer import *

dataset.load_dataset()
dataset.generate_batches(config.BLOCK_SIZE)

# train and test splits
DATA = dataset.batch_data
N = int(0.98 * len(DATA))
VALIDATION_INTERVAL = 10
OPTIMIZER_CHECKPOINT_NAME = "optimizer_cache"
MIN_LR = 0.00004
GRAD_MAX_NORM = 1.0
MAX_WARMUP_STEPS = 50
CHECKPOINT_INTERVAL = 15
SHOULD_WARMUP = True

val_loss_history = []
training_loss_history = []


def main():
    global MIN_LR, SHOULD_WARMUP
    args: HyperparamArgs = parse_arguments()
    logger.special(args)

    if args.dropout:
        MIN_LR = 0.00002

    if args.device.startswith("xla"):
        import torch_xla as xm
        args.device = xm.core.xla_model.xla_device()

    random.shuffle(DATA)
    train_data = DataLoader(
        DATA[:N], batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_data = DataLoader(DATA[N:], batch_size=args.batch_size, shuffle=True)

    # model with hyperparams
    runner = TransientRunner(
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=args.dropout,
        batch_size=args.batch_size,
    )
    runner.to_device(args.device)

    if args.compile:
        runner.compile_model()

    pretrained_model_exists = os.path.exists(args.from_pretrained)
    if pretrained_model_exists:
        runner = TransientRunner.load_from_checkpoint(args.from_pretrained)

    runner.use_parallel_if_available()
    runner.model.train()

    # print the number of parameters in the model
    logger.info(
        sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
        // 1_000_000,
        "M parameters",
    )
    optimizer = torch.optim.AdamW(
        runner.model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1
    )
    if os.path.exists(args.from_pretrained) and pretrained_model_exists:
        SHOULD_WARMUP = False
        optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_NAME))

    if args.in_jupyter:
        tqdm_notebook(range(args.epochs))
    else:
        tqdm(range(args.epochs))

    n_steps_per_batch = len(train_data) // args.gradient_acc
    n_steps = args.epochs * (n_steps_per_batch)
    if args.in_jupyter:
        tqdm_notebook(range(args.epochs))
    else:
        tqdm(range(args.epochs))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=VALIDATION_INTERVAL,
        limit_val_batches=1,
        accumulate_grad_batches=args.gradient_acc,
        enable_checkpointing=True,
        accelerator=args.device,
        precision="32"
    )
    trainer.fit(runner, train_dataloaders=train_data, val_dataloaders=val_data)

    runner.save(args.save_to, verbose=True)
    torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_NAME)
    show_loss(n_steps)


def set_lr(optimizer, new_lr):
    optimizer.param_groups[0]["lr"] = new_lr


def show_loss(epochs):
    """Display training and validation loss history on graph"""

    epoch_l = list(range(1, epochs + 1))
    plt.plot(epoch_l, training_loss_history, label="Training loss")
    plt.plot(epoch_l, val_loss_history, label="Validation loss")
    plt.legend()
    plt.savefig("loss_history.png")


class FakeGradScaler:
    """A placeholder GradScaler for when mixed precision is not available"""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return


if __name__ == "__main__":
    main()
