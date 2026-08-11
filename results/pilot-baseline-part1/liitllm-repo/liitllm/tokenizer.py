"""Byte-level BPE tokenizer, trained here on our own corpus.

Why byte-level BPE and not char-level: a char tokenizer needs ~4x more tokens
per document, so a fixed context sees a quarter as much text and every
training step buys less. BPE learns "ang ", "nag-", " naman" as single tokens.

Why *byte*-level specifically: the base alphabet is the 256 byte values, so any
UTF-8 text is representable and nothing can produce an unknown token. That is
what keeps this language-agnostic — and it is what let this project inherit
Munti's tokenizer unchanged: swap the corpus, retrain the merges, no code
changes and no English assumptions left behind.

Taglish is the reason the merges matter here. The corpus mixes Tagalog
morphology (nag-, pag-, -in-, reduplication) with English loanwords in the same
sentence, so the merge table has to cover both without a separate vocabulary for
either. Check compression_ratio before trusting a vocab size picked for English.

The `tokenizers` library provides the BPE merge algorithm (PRD FR-3 permits
this). We train the merges on our own Taglish corpus — we never download a
pretrained tokenizer.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# End-of-document marker. Documents are concatenated into one long stream, so
# the model needs an explicit signal for "this document is over" — otherwise it
# learns to run one straight into the next and never stops.
EOS = "<|endofdoc|>"


def train(
    texts,
    vocab_size: int = 4096,
    out_path: str | Path = "data/tokenizer.json",
) -> Tokenizer:
    """Train a byte-level BPE on `texts` (an iterable of strings) and save it."""
    tok = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space: treat "Once" at the start of a text the same as " Once"
    # mid-text, so the same word doesn't get two unrelated token sequences.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load(path: str | Path = "data/tokenizer.json") -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Run: python -m liitllm.data prepare"
        )
    return Tokenizer.from_file(str(path))


def compression_ratio(tok: Tokenizer, texts) -> float:
    """Characters per token — the number that justifies BPE over char-level."""
    chars = tokens = 0
    for t in texts:
        chars += len(t)
        tokens += len(tok.encode(t).ids)
    return chars / max(tokens, 1)
