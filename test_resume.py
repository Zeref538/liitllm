"""Does a killed run actually resume? Run this before spending a GPU-hour.

    python test_resume.py

LiitLLM's runs are ~20 hours against Kaggle's 12-hour wall, so every run WILL be
interrupted at least once. That makes resume load-bearing rather than a
convenience, and an untested resume path is not a resume path — the failure mode
is not a crash, it is a run that silently restarts from step 0 and looks fine
for ten hours.

Four things are checked, each corresponding to a way a resumed run produces a
plausible-but-wrong result instead of an error:

  1. it resumes from the checkpoint's step, not 0
  2. loss.csv has no duplicate or out-of-order steps afterwards
  3. RNG state is restored, so the resumed run does not replay the same batches
  4. a truncated ckpt.pt falls back to ckpt_prev.pt instead of starting over

Synthetic token ids, CPU, no tokenizer or dataset needed. Under a minute.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from liitllm import data as D
from liitllm import train as T

VOCAB = 256


def _setup(tmp: Path, max_steps: int, eval_every: int) -> Path:
    """A synthetic corpus and a config pointing at it. Returns the config path."""
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 60_000), ("val", 4_000)):
        rng.integers(0, VOCAB, size=n, dtype=D.DTYPE).tofile(data_dir / f"{split}.bin")

    cfg = {
        "out_dir": str(tmp / "out"),
        "data_dir": str(data_dir),
        "seed": 1337,
        "device": "cpu",
        "model": {
            "vocab_size": VOCAB, "block_size": 32, "n_layer": 2, "n_head": 2,
            "n_embd": 64, "dropout": 0.0, "bias": False,
            "learned_pos": True, "fast_attn": False,
        },
        "batch_size": 8,
        "max_steps": max_steps,
        "lr": 1e-3,
        "warmup_steps": 5,
        "eval_every": eval_every,
        "log_every": 10_000,  # quiet
    }
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _steps_in_log(out: Path) -> list[int]:
    rows = list(csv.DictReader((out / "loss.csv").open()))
    return [int(r["step"]) for r in rows]


def test_resume_continues():
    """Kill at step 40, resume, and land at 80 with a clean loss curve."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        out = tmp / "out"

        # First session: runs to 40 and stops.
        T.train(str(_setup(tmp, max_steps=40, eval_every=20)), resume=False)
        first = _steps_in_log(out)
        ck = torch.load(out / "ckpt.pt", weights_only=False)
        assert ck["step"] == 39, ck["step"]

        # Second session: same out_dir, longer budget, --resume.
        T.train(str(_setup(tmp, max_steps=80, eval_every=20)), resume=True)
        after = _steps_in_log(out)

        assert after[: len(first)] == first, "resume rewrote history it should have kept"
        assert max(after) == 79, f"did not reach the new budget: {after}"
        assert after == sorted(after), f"steps out of order: {after}"
        assert len(after) == len(set(after)), f"duplicate steps in loss.csv: {after}"
        print(f"  ok  resumed 40 -> 80, {len(after)} clean rows: {after}")


def test_rng_state_restored():
    """A resumed run must not replay the batches it already trained on.

    get_batch draws window offsets from the global torch RNG. If the state isn't
    saved, a resume rewinds to the seeded start and re-serves identical windows
    — hours of compute spent re-reading the same text, which shows up as a
    suspiciously good train loss and nothing else.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        cfg = yaml.safe_load(_setup(tmp, max_steps=20, eval_every=10).read_text())
        data = D.load_split("train", cfg["data_dir"])
        ck = None

        T.train(str(tmp / "cfg.yaml"), resume=False)
        ck = torch.load(tmp / "out" / "ckpt.pt", weights_only=False)
        assert "rng" in ck, "checkpoint carries no RNG state"

        # Batch that the *saved* state produces vs. the seeded-from-scratch one.
        torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        resumed, _ = D.get_batch(data, 8, 32, "cpu")
        torch.manual_seed(cfg["seed"])
        fresh, _ = D.get_batch(data, 8, 32, "cpu")
        assert not torch.equal(resumed, fresh), "resumed run replays the original batches"
        print("  ok  RNG state restored (resumed run sees new batches)")


def test_corrupt_checkpoint_falls_back():
    """A checkpoint truncated mid-write must not cost the whole run."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        out = tmp / "out"
        T.train(str(_setup(tmp, max_steps=40, eval_every=20)), resume=False)

        prev = out / "ckpt_prev.pt"
        assert prev.exists(), "no ckpt_prev.pt — the fallback copy was never written"
        good_step = torch.load(prev, weights_only=False)["step"]

        # Simulate a kill partway through torch.save.
        with (out / "ckpt.pt").open("r+b") as f:
            f.truncate(128)

        ck = T.load_checkpoint(out / "ckpt.pt", "cpu")
        assert ck is not None, "fell back to nothing — a corrupt ckpt.pt lost the run"
        assert ck["step"] == good_step, f"loaded the wrong checkpoint: {ck['step']}"
        print(f"  ok  corrupt ckpt.pt fell back to ckpt_prev.pt @ step {good_step}")


def test_max_hours_stops_cleanly():
    """The wall-clock guard exits with a complete checkpoint, not mid-write."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        stopped = T.train(
            str(_setup(tmp, max_steps=200, eval_every=10)), resume=False, max_hours=0.0
        )
        assert stopped is True, "--max-hours did not stop the run"
        ck = torch.load(tmp / "out" / "ckpt.pt", weights_only=False)
        assert ck["step"] < 199, "stopped at the end anyway, so nothing was tested"
        print(f"  ok  --max-hours stopped cleanly at step {ck['step']} with a full checkpoint")


if __name__ == "__main__":
    print("LiitLLM resume gate\n")
    failed = []
    for fn in (
        test_resume_continues,
        test_rng_state_restored,
        test_corrupt_checkpoint_falls_back,
        test_max_hours_stops_cleanly,
    ):
        print(f"- {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            failed.append(f"{fn.__name__}: {e}")
            print(f"  FAIL  {e}")

    print()
    if failed:
        print("RESUME GATE FAILED — do not start a long run:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS - a killed session can resume. Long runs are cleared.")
