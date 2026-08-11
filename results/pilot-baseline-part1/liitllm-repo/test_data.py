"""Does the data pipeline produce a valid ablation? Run before the CPU prep session.

    python test_data.py

Synthetic documents, no network, a few seconds. It checks the three things that
would silently invalidate the experiment rather than crash it:

  1. both corpora end up at exactly the same token count — otherwise the ablation
     measures data quantity instead of data quality, which looks like a result
  2. the filtered corpus really is the Taglish subset, not just a smaller sample
  3. the train/val split is carved and non-empty

The failure mode being guarded against is a pipeline that runs cleanly, produces
plausible .bin files, trains for 40 GPU-hours, and answers the wrong question.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from liitllm import data as D
from liitllm import taglish as tg

# Three document types, interleaved the way a real crawl mixes them.
TAGLISH = (
    "Grabe, nag-commute ako kanina from Makati papuntang office tapos na-late pa "
    "rin ako sa meeting. Ang traffic talaga sobrang hassle, kaya next time mag-book "
    "na lang siguro ako ng grab kesa mag-bus pa sa EDSA every morning."
)
TAGALOG = (
    "Ang mga bata ay naglalaro sa labas ng bahay tuwing hapon. Masaya sila dahil "
    "maganda ang panahon at walang ulan. Sabi ng kanilang nanay, kailangan nilang "
    "umuwi bago dumilim ang langit sa gabi at bago sila kumain ng hapunan."
)
ENGLISH = (
    "The company announced today that it will open a new office in the city next "
    "year. The plan is expected to create more jobs for local people and the "
    "government said it would support the project with additional funding."
)


def _fake_stream(_source):
    """Stand in for the HF streaming loader: 1 Taglish doc per 3, all unique."""
    for i in range(3000):
        base = (TAGLISH, TAGALOG, ENGLISH)[i % 3]
        yield f"{base} Document number {i} unique suffix."


def test_pipeline():
    D._stream = _fake_stream  # no network, deterministic content
    target = 60_000

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        data_dir = Path(d)
        D.prepare(target=target, vocab_size=400, tok_sample=600,
                  sources=("cc100",), data_dir=data_dir)

        sizes = {}
        for arm in ("filtered", "unfiltered"):
            train = np.memmap(data_dir / arm / "train.bin", dtype=D.DTYPE, mode="r")
            val = np.memmap(data_dir / arm / "val.bin", dtype=D.DTYPE, mode="r")
            assert len(train) > 0, f"{arm}/train.bin is empty"
            assert len(val) > 0, f"{arm}/val.bin is empty — no held-out data to eval on"
            sizes[arm] = len(train) + len(val)
            del train, val

        assert sizes["filtered"] == sizes["unfiltered"], (
            f"budget mismatch: filtered={sizes['filtered']:,} "
            f"unfiltered={sizes['unfiltered']:,} — the ablation would be comparing "
            "corpus SIZE, not corpus content"
        )
        print(f"  ok  both corpora at exactly {sizes['filtered']:,} tokens")
        print(f"  ok  val split carved from both arms")


def test_filter_selects_taglish():
    """The filtered arm must be the Taglish subset, not an arbitrary sample."""
    docs = [TAGLISH, TAGALOG, ENGLISH] * 10
    kept = [d for d in docs if tg.is_taglish(d)]
    assert len(kept) == 10, f"expected 10 Taglish docs kept, got {len(kept)}"
    assert all(d is TAGLISH for d in kept), "filter kept a non-Taglish document"
    print(f"  ok  filter kept {len(kept)}/{len(docs)} docs, all of them Taglish")


def test_dedup_drops_repeats():
    """Web crawls repeat heavily; duplicates are memorised first at 3 epochs."""
    docs = [TAGLISH, TAGLISH, TAGALOG, TAGLISH, ENGLISH]
    assert len(list(D._deduped(docs))) == 3, "duplicates survived dedup"
    print("  ok  duplicate documents dropped")


if __name__ == "__main__":
    print("LiitLLM data gate\n")
    failed = []
    for fn in (test_filter_selects_taglish, test_dedup_drops_repeats, test_pipeline):
        print(f"- {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            failed.append(f"{fn.__name__}: {e}")
            print(f"  FAIL  {e}")

    print()
    if failed:
        print("DATA GATE FAILED - the ablation would not be valid:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS - filtered and unfiltered corpora are a fair comparison.")
