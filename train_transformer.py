from tqdm import tqdm

from bigram_transformer import *


batch_size = 32
learning_rate = 0.0003
epochs = 2000
transformer_model_name = 'Bigram-Transformer.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if mps.is_built():
    device = torch.device('mps')


model = BigramLanguageModel().to_device(device)
model.load(transformer_model_name)


@torch.no_grad()
def estimate_loss():
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

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# first 90% will be train, rest validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    return x, y


# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) // 1_000_000, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in (t := tqdm(range(epochs))):
    # every once in a while evaluate the loss on train and val sets
    # if iter % eval_interval == 0 or iter == epochs - 1:
    #     losses = estimate_loss()
    #     t.set_description(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    t.set_description(f"Epoch {iter}: Train loss {loss:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.save('Bigram-Transformer.pt')
