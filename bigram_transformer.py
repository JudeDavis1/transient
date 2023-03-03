import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import BookCorpusDataset


# torch.manual_seed(1337)


# max content length for predictions
block_size = 64
eval_interval = 500
eval_iters = 1
n_embd = 384
n_layers = 8
n_head = 6
dropout = 0.1


dataset = BookCorpusDataset()
text = dataset.file_contents.split(' ')

# unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers and vice-versa
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        # default to CPU
        self.device = torch.device('cpu')
        self.transformer_model_name = 'Bigram-Transformer.pt'
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
         
         # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = token_embed + pos_embed # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    def generate(self, idx: torch.Tensor, max_new_tokens, display=False):
        cpu_dev = torch.device('cpu')

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if display:
                scalar_idx = idx_next.flatten().to(cpu_dev).tolist()
                sys.stdout.write(decode(scalar_idx))
                sys.stdout.flush()
        
        if display: print()
        
        return idx
    
    def to_device(self, device: torch.device):
        self.device = device
        print(f'Using {str(device).upper()} backend...')
        
        return self.to(device)
    
    def load(self, path='Bigram-Transformer.pt', **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))
    
    def save(self, path='Bigram-Transformer.pt'):
        torch.save(self.state_dict(), path)


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out


class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()

        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.LayerNorm(4 * n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.ffwd(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x




