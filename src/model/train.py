import os
import random

import torch
from matplotlib import pyplot as plt
from torch.backends import mps
from torch.cuda.amp import GradScaler
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
    runner.to_device(DEVICE)

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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.999, step_size=20)
    scaler = (
        GradScaler()
        if args.use_mixed_precision and DEVICE == "cuda"
        else FakeGradScaler()
    )


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
    )
    trainer.fit(runner, train_dataloaders=train_data, val_dataloaders=val_data)

    # completed_updates = 0
    # for iter in t:
    #     for j, (xb, yb) in enumerate(train_data):
    #         if scheduler.get_lr()[-1] < MIN_LR:
    #             set_lr(optimizer, MIN_LR)

    #         cur_step = (iter * len(train_data)) + j
    #         if SHOULD_WARMUP and cur_step <= MAX_WARMUP_STEPS:
    #             new_lr = (args.lr / MAX_WARMUP_STEPS) * cur_step
    #             set_lr(optimizer, new_lr)

    #         xb = xb.to(DEVICE)
    #         yb = yb.to(DEVICE)

    #         # with mixed precision
    #         with autocast(enabled=args.use_mixed_precision and DEVICE == "cuda"):
    #             if completed_updates % VALIDATION_INTERVAL == 0:
    #                 val_loss = get_val_loss(runner.model, val_data, eval_iters=2)

    #             # evaluate the loss
    #             _, loss = runner.forward(xb, yb)
    #             loss: torch.Tensor = loss / args.gradient_acc

    #         scaler.scale(loss.mean()).backward()
    #         nn.utils.clip_grad.clip_grad_norm_(
    #             runner.model.parameters(), max_norm=GRAD_MAX_NORM
    #         )

    #         total_loss += loss.mean().item()
    #         scheduler.step()

    #         if (cur_step + 1) % args.gradient_acc == 0 or (cur_step + 1) == n_steps:
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad(set_to_none=True)

    #             val_loss_history.append(val_loss)
    #             training_loss_history.append(total_loss)

    #             val_loss_str = round(val_loss, 6) if val_loss else "N/A"
    #             lr_str = scheduler.get_lr()[-1]

    #             update_num = (j // args.gradient_acc) + 1
    #             t.set_description(
    #                 f"Epoch {iter} - Batch: {update_num}/{n_steps_per_batch} - Train loss: {total_loss:.6f}  Validation loss: {val_loss_str}  LR: {lr_str:.7f}"
    #             )
    #             total_loss = 0
    #             completed_updates += 1

    #             if completed_updates % CHECKPOINT_INTERVAL == 0:
    #                 if DEVICE == "cuda":
    #                     torch.cuda.empty_cache()
    #                 runner.save(args.save_to, verbose=True)

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


@torch.no_grad()
def get_val_loss(model: TransformerModel, dataloader, eval_iters=50) -> float:
    """Estimates the validation loss of current model"""

    model.eval()

    val_loss = 0.0
    for _ in range(eval_iters):
        X, Y = get_batch(dataloader)

        _, loss = model(
            idx=X.to(DEVICE), targets=Y.to(DEVICE), start_pos=0, device=DEVICE
        )
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


if __name__ == "__main__":
    main()
