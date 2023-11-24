"""

All relevant modules for the transformer architecture.

"""

import os
import sys
from typing import Optional

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import functional as F

from src.model.rope import apply_rotary_emb, precompute_freqs_cis, repeat_kv

"""Local"""
from src import logger
from src.config import config
from src.dataset.dataset import BookCorpusDataset

dataset = BookCorpusDataset(folder="data")

# unique characters that occur in this text
tokens = dataset.corpus
vocab_size = dataset.vocab_size
logger.info("Vocab size:", vocab_size)


class TransientRunner(pl.LightningModule):
    def __init__(
        self,
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=0.2,
        batch_size=1,
    ):
        super().__init__()

        # hyperparams
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # default to CPU
        self.m_device = "cpu"
        self.cache_dir = "./model_cache"
        self.transformer_model_name = f"./models/BT-{n_heads}Head-{n_layers}Layer.pt"

        self.model = TransformerModel(
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            vocab_size=dataset.vocab_size,
            n_heads=self.n_heads,
            dropout=self.dropout,
            batch_size=batch_size,
        ).apply(self._init_weights)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor = None, start_pos=0
    ) -> torch.Tensor:
        return self.model(x, targets=targets, start_pos=start_pos, device=self.m_device)

    def use_parallel_if_available(self):
        """Use multiple GPUs if available"""

        if torch.cuda.device_count() > 1:
            logger.info("Using", torch.cuda.device_count(), "GPUs...")
            self.model = nn.DataParallel(self.model)

    def compile_model(self):
        """Compile the model for training"""
        self.model = torch.jit.script(self.model)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            self.model.parameters(), lr=0.0003, betas=(0.9, 0.98), weight_decay=0.1
        )

    def training_step(self, batch) -> STEP_OUTPUT:
        xb, yb = batch
        _, loss = self.forward(xb, yb)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch) -> STEP_OUTPUT:
        self.eval()
        xb, yb = batch
        _, loss = self.model(idx=xb, targets=yb, start_pos=0, device=self.device)
        self.log("val_loss", loss)
        self.train()

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: int = 0.5,
        display=False,
        greedy=True,
    ):
        """Generate new tokens from the model iteratively"""

        start_pos = idx.size(1)
        for _ in range(max_new_tokens):
            # Slice idx to get the current window (idx_cond)
            if idx.size(1) < self.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.block_size :]

            logits, _ = self.model(idx_cond, start_pos=start_pos, device=self.m_device)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits / temperature, dim=-1)  # (B, C)

            # sample from the distribution
            if greedy:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1).long()

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            start_pos += 1

            if display:
                scalar_idx = idx_next.flatten().cpu().numpy()
                sys.stdout.write(dataset.decode(scalar_idx))
                sys.stdout.flush()

        if display:
            print()

        return idx

    def is_parallel(self):
        return isinstance(self.model, nn.DataParallel)

    def to_device(self, device: str):
        self.m_device = device
        self.model = self.model.to(device)
        logger.info(f"Using {str(device).upper()} backend...")

    def load(self, load_cache=None, **kwargs):
        logger.info("[*] Loading model:", self.transformer_model_name)

        if os.path.exists(load_cache):
            # load the uncompressed copy
            self.model.load_state_dict(torch.load(load_cache, **kwargs))

    def save(self, name="model_cache", verbose=False):
        if verbose:
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
        vocab_size=1000,
        n_heads=8,
        dropout=0.2,
        batch_size=1,
    ):
        super().__init__()

        # hyperparams
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # each token directly reads off the logits for the next token from a lookup table
        self.token_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_heads=n_heads,
                    dropout=self.dropout,
                    batch_size=batch_size,
                )
                for _ in range(n_layers)
            ]
        )

        # final layer norm
        self.ln_f = RMSNorm(n_embd)
        self.dec_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.n_embd // self.n_heads, config.BLOCK_SIZE * 2
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        device: str = "cpu",
    ):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embed: torch.Tensor = self.token_table(idx.long()).to(device)  # (B, T, C)
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        mask = None
        # if T > 1:
        #     mask = torch.full(
        #         (1, 1, T, T), float("-inf"), device=idx.device
        #     )
        #     mask = torch.triu(mask, diagonal=start_pos if self.training else start_pos + 1).type_as(token_embed)
        #     # print(mask); exit(0)

        x: torch.Tensor = token_embed
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, start_pos=start_pos, mask=mask)
        
        x: torch.Tensor = self.dec_dropout(self.ln_f(x))  # (B, T, C)
        logits: torch.Tensor = self.lm_head(x)  # (B, T, dataset.vocab_size)

        if torch.jit.isinstance(targets, torch.Tensor):
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
            
        else:
            loss = None

        return logits, loss


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_heads, dropout, batch_size):
        # n_embd: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()

        head_size = n_embd // n_heads
        self.sa_block = MultiHeadAttention(
            n_embd, head_size, n_heads, dropout, batch_size=batch_size
        )
        self.ffwd = FeedForward(dim=n_embd, multiple_of=256, hidden_dim=n_embd * 4)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.block_dropout = nn.Dropout(dropout)

    def forward(
        self, x, freqs_cis: torch.Tensor, start_pos: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Add residual connections around each block"""
        x = x + self.block_dropout(
            self.sa_block(
                self.ln1(x), freqs_cis=freqs_cis, start_pos=start_pos, mask=mask
            )
        )
        x = x + self.block_dropout(self.ffwd(self.ln2(x)))

        return x


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, n_embd, head_size, n_heads, dropout, batch_size):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.dropout_p = dropout
        self.n_embd = n_embd

        self.dropout = nn.Dropout(dropout)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        head_dim = n_embd // n_heads
        self.cache_k = torch.zeros(
            (
                batch_size,
                config.BLOCK_SIZE,
                n_heads,
                head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                batch_size,
                config.BLOCK_SIZE,
                n_heads,
                head_dim,
            )
        )

    def forward(
        self, x, freqs_cis: torch.Tensor, start_pos: int, mask: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q: torch.Tensor = q.view(B, T, self.n_heads, self.head_size)
        k: torch.Tensor = k.view(B, T, self.n_heads, self.head_size)
        v: torch.Tensor = v.view(B, T, self.n_heads, self.head_size)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(v)

        self.cache_k[:B, start_pos : start_pos + T] = k.detach()
        self.cache_v[:B, start_pos : start_pos + T] = v.detach()

        k = self.cache_k[:B, : start_pos + T]
        v = self.cache_v[:B, : start_pos + T]

        # k = repeat_kv(k, 1)
        # v = repeat_kv(v, 1)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        dropout_p = self.dropout_p
        if not self.training:
            dropout_p = 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True
        )

        # join heads concatenating along the last dimension
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
