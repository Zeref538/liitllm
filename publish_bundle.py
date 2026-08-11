"""Build a public Kaggle dataset holding every result, on one account.

    python publish_bundle.py            # build only
    python publish_bundle.py --push     # build, then create/version the dataset

The four training runs live on two Kaggle accounts, and a kernel cannot be moved
between accounts — re-running seed 2 on the main account would cost another ~22
GPU-hours for artifacts that already exist. This bundles the outputs instead, so
one public dataset on the main account carries the whole experiment.

Checkpoints are saved weights-only: the optimizer state is ~2/3 of each file and
is useless to anyone who is not resuming that exact run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / ".publish"
OWNER = "johnandreimartinez"
SLUG = "liitllm-taglish"

# The run that reached step 60000 for each arm, and the label it publishes under.
RUNS = {
    "filtered-seed1":   "baseline-part1",
    "filtered-seed2":   "baseline2-part1",
    "unfiltered-seed1": "ablation-part2",
    "unfiltered-seed2": "ablation2-part2",
}


def strip_optimizer(src: Path, dst: Path) -> tuple[int, int]:
    """Copy a checkpoint without optimizer/scaler/RNG state."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    lean = {k: ck[k] for k in ("cfg", "model", "step", "val_loss") if k in ck}
    torch.save(lean, dst)
    return src.stat().st_size, dst.stat().st_size


def build() -> Path:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()

    for label, slug in RUNS.items():
        out = BUNDLE / label
        out.mkdir()
        src = HERE / "results" / slug / "out"
        assert src.exists(), f"missing run output: {src}"
        big, small = strip_optimizer(src / "ckpt.pt", out / "ckpt.pt")
        print(f"{label:18} {slug:18} {big / 1e6:6.0f}MB -> {small / 1e6:5.0f}MB")
        for name in ("loss.csv", "curve.png", "samples.md"):
            if (src / name).exists():
                shutil.copy(src / name, out / name)

    shutil.copytree(HERE / "results" / "verdict", BUNDLE / "verdict")
    shutil.copy(HERE / "results" / "00-prep" / "data" / "tokenizer.json", BUNDLE)
    shutil.copytree(HERE / "data" / "lexicons", BUNDLE / "lexicons")
    for doc in ("README.md", "CASE_STUDY.md"):
        shutil.copy(HERE / doc, BUNDLE)

    (BUNDLE / "dataset-metadata.json").write_text(json.dumps({
        "title": "LiitLLM — a Taglish language model, from scratch",
        "id": f"{OWNER}/{SLUG}",
        "licenses": [{"name": "MIT"}],
    }, indent=1), encoding="utf-8")
    return BUNDLE


def push():
    def kaggle(*a):
        return subprocess.run([sys.executable, "-m", "kaggle", *a],
                              capture_output=True, text=True, encoding="utf-8")

    who = kaggle("config", "view").stdout
    assert OWNER in who, f"wrong account live — expected {OWNER}:\n{who}"

    exists = "ready" in (kaggle("datasets", "status", f"{OWNER}/{SLUG}").stdout or "").lower()
    if exists:
        # -r zip is not optional: without it every subdirectory is silently dropped.
        r = kaggle("datasets", "version", "-p", str(BUNDLE), "-r", "zip",
                   "-m", "results for all four runs")
    else:
        r = kaggle("datasets", "create", "-p", str(BUNDLE), "-r", "zip", "-u")
    print(r.stdout or r.stderr)
    print(f"https://www.kaggle.com/datasets/{OWNER}/{SLUG}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true")
    a = p.parse_args()
    d = build()
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"\nbundle: {total / 1e6:.0f}MB at {d}")
    if a.push:
        push()
