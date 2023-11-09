"""

All relevant modules for the transformer architecture.

"""

import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

"""Local"""
from src import logger
from src.config import Config
from src.dataset.dataset import BookCorpusDataset

dataset = BookCorpusDataset(folder="data")

# unique characters that occur in this text
tokens = dataset.corpus
vocab_size = dataset.vocab_size
logger.info("Vocab size:", vocab_size)

class TransientRunner:
    def __init__(
        self,
        block_size=Config.BLOCK_SIZE,
        n_embd=Config.N_EMBD,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        dropout=0.2,
    ):
        # hyperparams
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # default to CPU
        self.device = torch.device("cpu")
        self.cache_dir = "./model_cache"
        self.transformer_model_name = f"./models/BT-{n_heads}Head-{n_layers}Layer.pt"

        self.model = TransformerModel(
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )

        # apply weights initialization
        self.model.apply(self._init_weights)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        return self.model(x, targets, device=self.device)

    def use_parallel_if_available(self):
        """Use multiple GPUs if available"""

        if torch.cuda.device_count() > 1:
            logger.info("Using", torch.cuda.device_count(), "GPUs...")
            self.model = nn.DataParallel(self.model)

    def compile_model(self):
        """Compile the model for training"""
        self.model = torch.compile(self.model)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens,
        temperature: int = 0.5,
        display=False,
        greedy=True,
    ):
        """Generate new tokens from the model iteratively"""

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self.model(idx_cond, device=self.device)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits / temperature, dim=-1)  # (B, C)

            # sample from the distribution
            if greedy:
                idx_next = torch.argmax(probs) \
                    .unsqueeze(-1) \
                    .unsqueeze(-1).long()
            else:
                idx_next = torch.multinomial(probs, num_samples=1).long()

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if display:
                scalar_idx = idx_next.flatten().cpu().numpy()

                sys.stdout.write(dataset.decode([int(scalar_idx[0])]))
                sys.stdout.flush()

        if display:
            print()

        return idx

    def score_accuracy(
        self,
        dataset: BookCorpusDataset,
        n_samples: int = 10,
        block_size: int = 3,
    ) -> float:
        """Score the accuracy of the model on a set of samples"""

        # generate a set of samples
        dataset.generate_batches(block_size)
        loader = DataLoader(dataset.prep_data[:n_samples], batch_size=1)

        correct = 0

        # iterate over the samples
        for x, y in (pbar := tqdm(loader, total=n_samples)):
            y = y.flatten()[:block_size].numpy()

            # generate predictions
            predictions = self.generate(x, block_size).flatten()[-block_size:].numpy()

            # compute accuracy
            correct += int(predictions.tolist() == y.tolist())
            pbar.set_description(f"Correct: {correct}")

        accuracy = correct / n_samples
        return accuracy

    def is_parallel(self):
        return isinstance(self.model, nn.DataParallel)

    def to_device(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device)
        logger.info(f"Using {str(device).upper()} backend...")

    def load(self, load_cache=None, **kwargs):
        logger.info("[*] Loading model:", self.transformer_model_name)

        if os.path.exists(load_cache):
            # load the uncompressed copy
            self.model.load_state_dict(torch.load(load_cache, **kwargs))

    def save(self, name="model_cache", save_cache=False):
        logger.info("[*] Saving model:", self.transformer_model_name)

        if self.is_parallel():
            self.model = self.model.module

        torch.save(self.model.state_dict(), name)

    def _init_weights(self, m: nn.Module):
        normal_dist = lambda param: nn.init.normal_(param, mean=0.0, std=0.02)

        if isinstance(m, nn.Linear):
            normal_dist(m.weight)
            if m.bias != None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            normal_dist(m.weight)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.in_channels
            std = (2.0 / n) ** 0.5
            m.weight.data.normal_(0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()


class TransformerModel(nn.Module):
    def __init__(
        self,
        block_size=128,
        n_embd=384,
        n_layers=8,
        n_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        # hyperparams
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # each token directly reads off the logits for the next token from a lookup table
        self.token_table = nn.Embedding(dataset.vocab_size, n_embd)
        self.position_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_heads=n_heads,
                    dropout=self.dropout,
                    block_size=self.block_size,
                )
                for _ in range(n_layers)
            ]
        )

        # final layer norm
        self.ln_f = RMSNorm(n_embd)
        self.dec_dropout = nn.Dropout(self.dropout)
        self.lm_head = nn.Linear(n_embd, dataset.vocab_size, bias=False)

    def forward(self, idx, targets=None, device=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embed = self.token_table(idx)  # (B, T, C)
        pos_embed = self.position_table(torch.arange(T, device=device))  # (T, C)
        x = self.dec_dropout(token_embed + pos_embed)  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.dec_dropout(self.ln_f(x))  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, dataset.vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_heads, block_size, dropout):
        # n_embd: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()

        head_size = n_embd // n_heads
        self.sa_block = MultiHeadAttention(
            n_embd, head_size, n_heads, block_size, dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.block_dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        """Add residual connections around each block"""
        x = x + self.block_dropout(self.sa_block(self.ln1(x)))
        x = x + self.block_dropout(self.ffwd(self.ln2(x)))

        return x


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, n_embd, head_size, n_heads, block_size, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.dropout_p = dropout
        self.flash = True
        self.n_embd = n_embd

        self.dropout = nn.Dropout(dropout)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size(0)
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(batch_size, T, self.n_heads, self.head_size)
        k = k.view(batch_size, T, self.n_heads, self.head_size)
        v = v.view(batch_size, T, self.n_heads, self.head_size)
        
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            dropout_p = self.dropout_p
            if not self.training:
                dropout_p = 0
            
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            # compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size**0.5)
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)

            # compute values and attention output
            out = torch.matmul(attn, v).view(batch_size, T, self.n_embd)

        # join heads concatenating along the last dimension
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.fd1 = nn.Linear(n_embd, 4 * n_embd)
        self.fd2 = nn.Linear(4 * n_embd, n_embd)
        self.ln = RMSNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.fd1(x)
        x = F.gelu(x)
        x = self.fd2(x)
        x = self.dropout(x)
        return x


class RMSNorm(torch.nn.Module):
    """Normalization which is the same as the one LLaMA uses"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
