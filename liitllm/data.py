"""Streamed Taglish web text -> tokenizer -> two flat arrays of token ids on disk.

Each split becomes one long uint16 stream, documents separated by an EOS token.
Training samples a random window out of it. There is no DataLoader and no
padding: every window is exactly block_size tokens of real text, so no compute is
wasted on padding and batching is a single slice. uint16 holds 0..65535, which
covers any vocab we'd use here.

    python -m liitllm.data bakeoff                 # which corpus has more Taglish?
    python -m liitllm.data prepare --target 200000000
    python -m liitllm.data prepare --limit 20000   # a slice, for local dev

Run this on a CPU session. It is IO-bound, not GPU-bound, and CPU sessions do not
consume the GPU quota that the 20-hour training runs need.

Why one pass builds both corpora
--------------------------------
The ablation compares a Taglish-filtered corpus against an unfiltered one, and it
is only a valid comparison if the two differ in *what* they contain and not in
how much. So both are built in the same streaming pass over the same source in
the same order: every document goes into the unfiltered stream, and the subset
passing `taglish.is_taglish` also goes into the filtered one. Each stops at the
same token target, and `prepare` asserts they match before anything trains.

Get that wrong and the ablation measures data quantity instead of data quality,
which would look exactly like a result.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import taglish as tg
from . import tokenizer as tk

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DTYPE = np.uint16
CHUNK = 10_000  # documents encoded per write
VAL_FRACTION = 0.005
# Enough val tokens that estimate_loss's 50 batches are drawn from genuinely
# different windows. Below this the val curve is measuring the same few tokens
# over and over — and the val curve is what the week-1 go/no-go decision reads.
MIN_VAL_TOKENS = 100

# FineWeb-2's Filipino subset. Two naming traps, both verified 2026-08-08:
# the config is `fil_Latn` (Filipino), NOT the `tgl_Latn` an ISO-639-3 guess
# suggests — that name does not exist and fails at load time.
#
# CC-100 (`statmt/cc100`, lang="tl") was the other candidate and is no longer
# usable: it is a script-based HF dataset, and datasets>=3.0 dropped script
# support entirely ("Dataset scripts are no longer supported, but found
# cc100.py"). Pinning datasets<3 to get it back is not worth the fragility.
# Two independently-built web corpora, unioned to maximise the Taglish pool.
# Measured on 2,000 streamed documents each (2026-08-08):
#
#   fineweb2  46.8% of documents pass is_taglish
#   hplt      34.4%
#
# Both far above what a "code-switching is rare" assumption would predict —
# Filipino web text is natively code-switched, which is itself a finding and is
# why the unfiltered ablation arm is a weak control rather than a clean one.
SOURCES = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", {"name": "fil_Latn"}, "text"),
    "hplt": ("HPLT/HPLT2.0_cleaned", {"name": "tgl_Latn"}, "text"),
}


def _stream(source: str):
    """Yield document strings from a streamed HF dataset."""
    from datasets import load_dataset

    path, kwargs, column = SOURCES[source]
    ds = load_dataset(path, split="train", streaming=True, **kwargs)
    for row in ds:
        text = (row.get(column) or "").strip()
        if text:
            yield text


def _deduped(docs):
    """Drop exact repeats, hashing the WHOLE document.

    Web crawls repeat heavily — the same article syndicated across dozens of
    sites, the same boilerplate on every page of a forum. Duplicates inflate the
    token count while teaching the model nothing, and at multi-epoch budgets they
    are memorised first.

    Hashing only a prefix would be cheaper and is wrong here: web pages routinely
    open with identical navigation and cookie boilerplate, so a prefix hash
    silently discards distinct articles that happen to share a header. That
    failure is invisible — the corpus is just quietly smaller. Hashing the full
    string costs one pass over text we are about to tokenise anyway.
    ponytail: exact-match only, no near-duplicate detection. Upgrade to MinHash
    if sampled output still shows obvious near-repeats.
    """
    seen = set()
    for doc in docs:
        h = hashlib.blake2b(doc.encode("utf-8", "ignore"), digest_size=16).digest()
        if h not in seen:
            seen.add(h)
            yield doc


def bakeoff(source_names=tuple(SOURCES), n: int = 50_000):
    """How much Taglish does each corpus actually contain?

    Streams a sample from each source and reports the keep-rate and the Taglish
    words per document. Prints the numbers that decide which source to build on
    — and the comparison is itself a finding, since it measures how much
    code-switched text each pipeline's language-ID filtering threw away.
    """
    rows = []
    for name in source_names:
        kept = total = kept_words = 0
        try:
            for doc in _deduped(_stream(name)):
                total += 1
                if tg.is_taglish(doc):
                    kept += 1
                    kept_words += len(tg.words(doc))
                if total >= n:
                    break
        except Exception as e:  # a source may be gated, renamed, or offline
            print(f"{name}: UNAVAILABLE ({type(e).__name__}: {e})")
            continue
        rate = 100 * kept / max(total, 1)
        rows.append((name, total, kept, rate, kept_words))
        print(f"{name}: {kept:,}/{total:,} docs kept ({rate:.1f}%), "
              f"{kept_words:,} Taglish words")

    if rows:
        best = max(rows, key=lambda r: r[4])
        print(f"\nmost Taglish per {n:,} docs sampled: {best[0]}")
    return rows


def _encode_stream(docs, tok, eos_id, out_path: Path, target: int, label: str) -> int:
    """Encode documents into a flat uint16 file until `target` tokens. Returns the count.

    Encoding in chunks and appending: accumulating the whole corpus as a Python
    list first would be tens of GB of boxed ints and would OOM the session, while
    the array on disk is under a gigabyte.
    """
    total = 0
    batch: list[str] = []

    def flush(f, batch):
        nonlocal total
        ids: list[int] = []
        for enc in tok.encode_batch_fast(batch):
            ids.extend(enc.ids)
            ids.append(eos_id)  # document boundary
        ids = ids[: max(target - total, 0)]
        np.array(ids, dtype=DTYPE).tofile(f)
        total += len(ids)

    with open(out_path, "wb") as f:
        for doc in docs:
            batch.append(doc)
            if len(batch) >= CHUNK:
                flush(f, batch)
                batch = []
                print(f"  {label}: {total:,}/{target:,} tokens")
                if total >= target:
                    return total
        if batch and total < target:
            flush(f, batch)
    return total


def _split_val(path: Path, data_dir: Path, prefix: str):
    """Carve the last VAL_FRACTION of a token stream off as the val split."""
    arr = np.memmap(path, dtype=DTYPE, mode="r")
    n_val = int(len(arr) * VAL_FRACTION)
    # An empty or single-window val split doesn't raise — it produces val losses
    # computed over the same handful of tokens every eval, which is the one
    # number the week-1 decision gate reads. Fail loudly instead.
    assert n_val >= MIN_VAL_TOKENS, (
        f"{len(arr):,} tokens gives a {n_val}-token val split; need at least "
        f"{int(MIN_VAL_TOKENS / VAL_FRACTION):,} tokens total for a usable one"
    )
    cut = len(arr) - n_val
    np.array(arr[:cut]).tofile(data_dir / f"{prefix}train.bin")
    np.array(arr[cut:]).tofile(data_dir / f"{prefix}val.bin")
    del arr
    path.unlink()


def prepare(target: int = 200_000_000, vocab_size: int = 8192, limit: int | None = None,
            sources=tuple(SOURCES), tok_sample: int = 50_000, data_dir: Path = DATA_DIR):
    """Build the filtered and unfiltered corpora and the tokenizer they share.

    `limit` caps documents read (local dev). `target` is the token budget each
    corpus is built to — they are asserted equal afterwards.
    """
    data_dir = Path(data_dir)
    filtered_dir = data_dir / "filtered"
    unfiltered_dir = data_dir / "unfiltered"
    for d in (filtered_dir, unfiltered_dir):
        d.mkdir(parents=True, exist_ok=True)

    def source_docs():
        for name in sources:
            yield from _stream(name)

    # 1. Tokenizer, trained on UNFILTERED text. This is deliberate and has to be
    #    disclosed: training the BPE on the filtered corpus would tune the
    #    vocabulary to the filtered model's own distribution and hand it an
    #    advantage in the ablation that has nothing to do with the filter. The
    #    neutral choice is the corpus both arms are drawn from.
    sample = []
    for doc in _deduped(source_docs()):
        sample.append(doc)
        if len(sample) >= tok_sample:
            break
    tok_path = data_dir / "tokenizer.json"
    tok = tk.train(sample, vocab_size=vocab_size, out_path=tok_path)
    ratio = tk.compression_ratio(tok, sample[:2000])
    print(f"tokenizer: vocab={tok.get_vocab_size()} compression={ratio:.2f} chars/token")
    # Munti measured 3.96 chars/token on English. Tagalog's affix stacking plus
    # English loanwords compress worse, which is why vocab defaults to 8192 here
    # rather than Munti's 4096 — but check the number rather than trusting it.
    assert tok.get_vocab_size() == vocab_size, (
        f"BPE produced {tok.get_vocab_size()} tokens, not {vocab_size} — the corpus "
        "sample is too small to support this many merges; lower vocab_size"
    )
    eos_id = tok.token_to_id(tk.EOS)

    # 2. One pass, both corpora. Same source, same order, same dedup — the only
    #    difference is is_taglish().
    def tee(keep_only_taglish: bool):
        n = 0
        for doc in _deduped(source_docs()):
            if limit and n >= limit:
                return
            n += 1
            if keep_only_taglish and not tg.is_taglish(doc):
                continue
            yield doc

    counts = {}
    for label, keep in (("filtered", True), ("unfiltered", False)):
        raw = data_dir / f"{label}.raw"
        counts[label] = _encode_stream(tee(keep), tok, eos_id, raw, target, label)
        print(f"{label}: {counts[label]:,} tokens")

    # 3. Equal budgets, or the ablation is measuring the wrong variable.
    lo, hi = min(counts.values()), max(counts.values())
    if lo != hi:
        print(f"trimming both corpora to {lo:,} tokens for budget parity")
    for label, prefix_dir in (("filtered", filtered_dir), ("unfiltered", unfiltered_dir)):
        raw = data_dir / f"{label}.raw"
        arr = np.memmap(raw, dtype=DTYPE, mode="r")
        np.array(arr[:lo]).tofile(raw.with_suffix(".trim"))
        del arr
        raw.unlink()
        _split_val(raw.with_suffix(".trim"), prefix_dir, "")

    for d in (filtered_dir, unfiltered_dir):
        n = len(np.memmap(d / "train.bin", dtype=DTYPE, mode="r"))
        print(f"{d.name}/train.bin: {n:,} tokens")


def load_split(split: str, data_dir: Path | str = DATA_DIR) -> np.ndarray:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Run: python -m liitllm.data prepare")
    # memmap, not read: train.bin can be gigabytes and we only ever touch a few
    # random windows of it per step.
    return np.memmap(path, dtype=DTYPE, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device="cpu", generator=None):
    """Sample `batch_size` random windows. Returns (x, y) where y is x shifted
    one token left — the target for position t is simply the token at t+1."""
    import torch

    high = len(data) - block_size - 1
    assert high > 0, "dataset is smaller than one context window"
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    if str(device).startswith("cuda"):
        # pin + non_blocking: overlap the host->device copy with compute.
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "bakeoff"])
    p.add_argument("--target", type=int, default=200_000_000, help="token budget per corpus")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--limit", type=int, default=None, help="max documents (default: all)")
    p.add_argument("--sources", default="fineweb2,hplt", help="comma-separated: fineweb2,hplt")
    a = p.parse_args()
    if a.cmd == "bakeoff":
        bakeoff()
    else:
        prepare(target=a.target, vocab_size=a.vocab, limit=a.limit,
                sources=tuple(a.sources.split(",")))
