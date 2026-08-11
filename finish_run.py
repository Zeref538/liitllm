"""Wait for the last training run, then produce the verdict. No input needed.

    python finish_run.py

Waits for ablation2-part1 to finish on Kaggle, downloads it, checks all four
checkpoints are real and equal-length, then scores them locally and writes
results/verdict/.

Why eval runs locally instead of on Kaggle: the four checkpoints are split
across two accounts, and a Kaggle kernel cannot mount another account's private
kernel output. Publishing them to share across accounts would be a bigger change
than scoring 33M-param models on a CPU, which is a one-off job of roughly an
hour and costs no quota. notebooks/99_eval.ipynb remains for a single-account
reproduction.

Safe to re-run at any point. Work already done is detected and skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

import run_pipeline as rp

HERE = Path(__file__).resolve().parent
TOKENIZER = HERE / "results" / "00-prep" / "data" / "tokenizer.json"
VERDICT = HERE / "results" / "verdict"

# The final checkpoint of each run, and the account that holds it. These are the
# parts that actually reached step 60000 — NOT a uniform "-part4", because each
# run needed a different number of continuations.
ARMS = {
    "baseline_filtered": [
        ("johnandreimartinez", "baseline-part1"),
        ("zeref824", "baseline2-part1"),
    ],
    "ablation_unfiltered": [
        ("johnandreimartinez", "ablation-part2"),
        ("zeref824", "ablation2-part1"),
    ],
}

PENDING = ("zeref824", "ablation2-part1")   # the one still training


def ckpt_of(slug: str) -> Path:
    return HERE / "results" / slug / "out" / "ckpt.pt"


def wait_and_fetch(user: str, slug: str) -> bool:
    """Block until the kernel finishes, then pull its output."""
    if rp.last_step(slug) >= 0:
        rp.log(f"{slug}: already downloaded")
        return True
    st = rp.wait_for(user, slug)
    if st != "complete":
        rp.log(f"{slug} ended '{st}' — NOT running eval. Check the kernel log.")
        return False
    rp.fetch_output(user, slug)
    return True


def main():
    rp.acquire_lock()

    user, slug = PENDING
    if not wait_and_fetch(user, slug):
        raise SystemExit(1)

    # A session can end before its target: Kaggle's wall, the --max-hours guard,
    # a crash. Seed 1's ablation did exactly this, stopping at 59000. Top it up
    # BEFORE comparing. run_arm is only safe to call now because the kernel has
    # already reached a terminal state above — calling it against a *running*
    # kernel would push a duplicate.
    goal = rp.target_steps("ablation2")
    if rp.last_step(slug) < goal:
        rp.log(f"{slug} at {rp.last_step(slug)}/{goal} — continuing before eval")
        if not rp.run_arm("ablation2", user, dry_run=False):
            raise SystemExit(f"{slug} could not be finished — not running eval")

    # Every arm must have trained the same number of steps. An ablation where one
    # arm saw more data measures the extra data, not the filter — so this refuses
    # to produce a verdict rather than producing a misleading one.
    steps, final = {}, {}
    for arm, runs in ARMS.items():
        for _, s in runs:
            run = s.rsplit("-part", 1)[0]
            # A continuation writes a NEW part dir, so the run's end is whichever
            # part got furthest — not the highest part number, which could be a
            # part that errored on startup and wrote nothing.
            parts = [(rp.last_step(p.parent.parent.name), p)
                     for p in (HERE / "results").glob(f"{run}-part*/out/ckpt.pt")]
            parts = [(st, p) for st, p in parts if st >= 0]
            if not parts:
                raise SystemExit(f"no usable checkpoint for {run}")
            st, path = max(parts)
            steps[path.parent.parent.name] = st
            final.setdefault(arm, []).append(str(path))
    rp.log(f"steps: {steps}")
    if len(set(steps.values())) != 1:
        raise SystemExit(
            f"runs did not train equally: {steps}\n"
            f"Top up the short one before comparing, or the gap is confounded."
        )

    if not TOKENIZER.exists():
        raise SystemExit(f"missing tokenizer: {TOKENIZER}")

    from liitllm.evaluate import compare
    VERDICT.mkdir(parents=True, exist_ok=True)
    rp.log(f"scoring 4 checkpoints on cpu -> {VERDICT} (expect ~1h)")
    compare(
        final["baseline_filtered"],
        final["ablation_unfiltered"],
        str(TOKENIZER),
        out_dir=str(VERDICT),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    rp.log("=== VERDICT WRITTEN ===")
    for f in sorted(VERDICT.iterdir()):
        rp.log(f"  {f.name}")


if __name__ == "__main__":
    main()
