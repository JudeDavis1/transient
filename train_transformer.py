import os
import sys
from tqdm import tqdm
from torch.backends import mps

import config

from bigram_transformer import *


batch_size = 128
learning_rate = 0.0004
val_interval = 10
epochs = int(sys.argv[1])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if mps.is_built():
    device = torch.device('mps')


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# model with hyperparams
model = BigramLanguageModel(
    block_size=config.BLOCK_SIZE,
    n_embd=config.N_EMBD,
    n_layers=config.N_LAYERS,
    n_head=config.N_HEAD,
    dropout=0.2
).to_device(device)


def main():
    if os.path.exists(model.transformer_model_name):
        print("Loading model:", model.transformer_model_name)
        model.load(model.transformer_model_name, map_location=device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) // 1_000_000, 'M parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t = tqdm(range(epochs))
    val_loss = 0
    for iter in t:
        optimizer.zero_grad(set_to_none=True)
        xb, yb = get_batch('train')

        if (iter + 1) % val_interval == 0:
            val_loss = get_val_loss(model, 5)

        # evaluate the loss
        _, loss = model(xb, yb)
        t.set_description(f"Epoch {iter} - Train loss: {loss:.4f}  Validation loss: {round(val_loss, 5) if val_loss else 'N/A'}")

        loss.backward()
        optimizer.step()

    model.save(model.transformer_model_name)



@torch.no_grad()
def get_val_loss(model: BigramLanguageModel, eval_iters=50):
    """Estimates the validation loss of current model"""

    model.eval()
    
    val_loss = 0
    for _ in range(eval_iters):
        X, Y = get_batch('val')

        _, loss = model(X, Y)
        val_loss += loss.item()
    
    # get the mean
    val_loss /= eval_iters
    model.train()
    
    return val_loss


def get_batch(split: str):
    """Generate a small batch of data of inputs x and targets y"""

    split = split.lower()
    assert split in ['val', 'train'], 'Split must be either \'val\' or \'train\''

    data = train_data if split == 'train' else val_data

    # subtracting block_size because we may go over len(data) for the y set
    max_idx = len(data) - model.block_size
    ix = torch.randint(max_idx, (batch_size,))
    x = torch.stack([data[i: i + model.block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + model.block_size] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    return x, y

if __name__ == '__main__':
    main()
