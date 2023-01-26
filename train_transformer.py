from bigram_transformer import *

from tqdm import tqdm


model = BigramLanguageModel().to(device)

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

# model.load_state_dict(torch.load(transformer_model_name, map_location=device))

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())//1000000, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in (t := tqdm(range(epochs))):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == epochs - 1:
        losses = estimate_loss()
        t.set_description(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), transformer_model_name)
