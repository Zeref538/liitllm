"""The transformer, written by hand.

A decoder-only transformer is a stack of identical blocks. Each block does two
things in sequence, each wrapped in a residual connection:

  1. Attention — every token looks at the tokens before it and pulls in what it
     needs. This is the only place information moves *between* positions.
  2. MLP — every token independently thinks about what it just gathered. No
     cross-token movement here at all; it's the same small network applied to
     each position separately.

Everything else (embeddings in, layernorms, a projection back out to vocabulary
logits) is plumbing around those two operations.

We use pre-LN (normalize *before* each sub-layer, not after). Post-LN sits on
the residual path and makes deep stacks need a careful warmup to train at all;
pre-LN leaves the residual stream a clean identity path from input to output,
which is why it's the modern default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class LiitConfig:
    vocab_size: int = 4096
    block_size: int = 256  # context length: how far back a token can attend
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # must be divisible by n_head
    dropout: float = 0.0
    bias: bool = False  # biases in Linear/LayerNorm; off is slightly better+faster
    # --- ablation switch (PRD FR-9) -------------------------------------
    # False removes positional information entirely, leaving attention
    # permutation-invariant: the model can see *which* tokens preceded it but
    # not in what order. Expect fluent-looking word salad.
    learned_pos: bool = True
    # --- speed switch ----------------------------------------------------
    # True uses PyTorch's fused attention kernel. Mathematically identical to
    # the manual path below (test_liitllm.py asserts this); the manual path stays
    # as the readable reference.
    fast_attn: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Each token emits a query ("what am I looking for?"), a key ("what am I?")
    and a value ("what will I hand over if picked"). Attention scores every
    query against every key, masks out the future, softmaxes into weights, and
    returns a weighted sum of values.

    "Multi-head" just means we split the embedding into `n_head` independent
    slices and run all of that separately in each, so different heads can
    specialize. It's one batched matmul, not a loop.
    """

    def __init__(self, cfg: LiitConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly into n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.fast_attn = cfg.fast_attn

        # One projection producing q, k and v at once — cheaper than three.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # The causal mask: a lower-triangular matrix of ones. Position i may
        # attend to j only where j <= i. Registered as a buffer so it moves to
        # the GPU with the model but isn't a learned parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): heads become a batch dimension.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.fast_attn:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # --- the reference implementation ---------------------------
            # Scale by 1/sqrt(head_dim): without it, dot products grow with
            # dimension, softmax saturates, and gradients vanish.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # -inf before softmax => exactly 0 weight on future tokens.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble the heads back into one embedding and mix them.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, non-linearity, project back.

    The 4x expansion is the standard transformer ratio — it gives the layer room
    to compute in a wider space before compressing back to the residual width.
    GELU rather than ReLU: smooth, and the empirical default for transformers.
    """

    def __init__(self, cfg: LiitConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: attention, then MLP, each with a residual."""

    def __init__(self, cfg: LiitConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # x = x + sublayer(norm(x)) — the residual stream is never normalized in
        # place, so gradients flow straight from the loss back to the embeddings.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Liit(nn.Module):
    def __init__(self, cfg: LiitConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        # Positional embeddings: attention itself is order-blind (it's a
        # weighted sum — permute the inputs and you permute the outputs). These
        # learned per-position vectors are the only thing telling the model
        # where a token sits. Turn them off to see that fall apart (ablation).
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.learned_pos else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the matrix mapping token->vector and the one mapping
        # vector->token-scores are the same matrix. Saves vocab*n_embd params
        # (a big share of a model this small) and consistently helps.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Residual projections get a smaller init, scaled by depth: every block
        # adds into the same residual stream, so without this the stream's
        # variance grows with n_layer and deep stacks start out unstable.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits, loss)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)  # broadcasts over the batch
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Next-token prediction: position t's logits are scored against
            # token t+1. The shift is done by the caller (see data.get_batch),
            # so here it's a straight flattened cross-entropy.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    # --- checkpointing -------------------------------------------------
    def checkpoint(self, **extra) -> dict:
        return {"cfg": asdict(self.cfg), "model": self.state_dict(), **extra}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device="cpu") -> "Liit":
        model = cls(LiitConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        return model.to(device)
