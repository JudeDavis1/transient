import os
import sys
from tqdm import tqdm
from torch.backends import mps

from bigram_transformer import *


batch_size = 128
learning_rate = 0.0003
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
    block_size=128,
    n_embd=384,
    n_layers=12,
    n_head=12,
    dropout=0.2
).to_device(device)


def main():
    if os.path.exists(model.transformer_model_name):
        print("Loading model:", model.transformer_model_name)
        model.load(model.transformer_model_name, map_location='cpu')

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) // 1_000_000, 'M parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t = tqdm(range(epochs))
    for iter in t:
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        t.set_description(f"Epoch {iter}: Train loss {loss:.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.save(model.transformer_model_name)



@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters=100):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            _, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    
    return out


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - model.block_size, (batch_size,))
    x = torch.stack([data[i: i + model.block_size] for i in ix])
    y = torch.stack([data[i + 1: i + model.block_size + 1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    return x, y

if __name__ == '__main__':
    main()