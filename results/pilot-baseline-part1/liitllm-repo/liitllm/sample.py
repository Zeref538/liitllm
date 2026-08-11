"""Sampling: turn the model's next-token distribution into text.

The model gives a probability for every token in the vocabulary. Sampling is
about how much of that distribution we're willing to draw from:

  temperature — divides the logits before softmax. <1 sharpens toward the
    likeliest token (safe, repetitive); >1 flattens (varied, incoherent);
    ->0 becomes greedy/argmax.
  top_k — keep only the k likeliest tokens, zero the rest. Cuts the long tail
    of garbage tokens that collectively hold real probability mass and would
    otherwise derail a generation every few hundred tokens.

    python -m liitllm.sample --ckpt out/ckpt.pt --prompt "Kahapon nag-"
"""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from . import tokenizer as tk
from .model import Liit


@torch.no_grad()
def generate(
    model: Liit,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 200,
    eos_id: int | None = None,
) -> torch.Tensor:
    """idx: (B, T) of context tokens. Appends max_new_tokens, autoregressively."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        # The model can only see block_size tokens, so crop to the most recent.
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # only the last position predicts the next token

        if temperature <= 0:  # greedy
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    if was_training:
        model.train()
    return idx


def generate_text(model, tok, prompt: str, device="cpu", **kw) -> str:
    # A tokenizer that doesn't match the checkpoint produces confident garbage
    # rather than an error, so check it once here — this is the only place a
    # tokenizer and a model ever meet.
    if tok.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tok.get_vocab_size()} != model vocab "
            f"{model.cfg.vocab_size} — this tokenizer belongs to a different run"
        )
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    if ids.numel() == 0:  # empty prompt: start from EOS as a "document begins" cue
        ids = torch.tensor([[tok.token_to_id(tk.EOS)]], dtype=torch.long, device=device)
    out = generate(model, ids, eos_id=tok.token_to_id(tk.EOS), **kw)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="Kahapon nag-")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    ckpt = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    model = Liit.from_checkpoint(ckpt, device=a.device)
    print(
        generate_text(
            model, tk.load(), a.prompt, device=a.device,
            max_new_tokens=a.tokens, temperature=a.temperature, top_k=a.top_k,
        )
    )
