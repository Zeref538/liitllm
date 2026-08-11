"""Generate the Kaggle notebooks from the cell sources below.

    python notebooks/build_notebooks.py

Notebooks are edited here, as plain Python, and the .ipynb files are build
output. Hand-editing .ipynb JSON means escaping every quote and newline in a
diff nobody can read; keeping the real source as Python costs ~20 lines of
scaffolding and makes the cells reviewable.

The notebooks themselves contain no logic — they mount paths, assert the
environment is what we think it is, and call into the package. Anything with a
branch in it belongs in liitllm/ where the tests can reach it.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Kaggle's default accelerator is a P100 (sm_60) that Kaggle's own PyTorch build
# does not support, and everything before the first training step is CPU work —
# so a doomed session looks healthy for ten minutes. Fail in the first cell.
GPU_ASSERT = '''
import torch, sys
assert torch.cuda.is_available(), "no GPU — set the accelerator to T4 x2"
cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
print(f"{name}  sm_{cap[0]}{cap[1]}")
assert cap >= (7, 0), (
    f"{name} is sm_{cap[0]}{cap[1]}; Kaggle's PyTorch needs sm_70+. "
    "Set --accelerator NvidiaTeslaT4 (the default P100 will not work)."
)
'''

# Every kernel output contains a full copy of the repo, so globbing for
# pyproject.toml anywhere under /kaggle/input finds stale snapshots from earlier
# runs. Resolve by dataset slug, then assert exactly one match.
FIND_REPO = '''
import shutil, os, sys

REPO_SLUG = "liitllm-repo"   # dataset holding this repo
PREP_SLUG = "00-prep"        # kernel whose OUTPUT holds the corpora + tokenizer

repo_src = find_one(REPO_SLUG, "pyproject.toml").parent
REPO = Path("/kaggle/working/liitllm-repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(repo_src, REPO)
sys.path.insert(0, str(REPO))
os.chdir(REPO)
print(f"repo: {repo_src} -> {REPO}")
'''

GATES = '''
# The correctness gates. Both are free (CPU, ~1 min) and they are the entire
# reason any bad Taglish output can be blamed on the data instead of the code.
# Never skip them to save a minute at the start of a 20-hour run.
!python test_liitllm.py
!python test_resume.py
'''

# Artifacts move between sessions as KERNEL OUTPUT, not as datasets.
#
# A pushed kernel's /kaggle/working is saved automatically and can be mounted by
# the next kernel via kernel_sources. That is how the prep session hands the
# corpora to training, and how each training part hands its checkpoint to the
# next. The alternative — versioning a dataset from inside the notebook — needs
# API credentials in notebook Secrets, can fail at the very end of a 4-hour run,
# and silently drops directories without `-r zip`. Mounting kernel output has
# none of those failure modes and no credentials at all.
MOUNT = '''
import glob
from pathlib import Path

def find_one(slug, filename):
    """Locate a mounted dataset or kernel output by slug, not a bare glob.

    Two things this has to get right:

    1. Kaggle mounts sources at /kaggle/input/datasets/<owner>/<slug>/... and
       /kaggle/input/kernels/<owner>/<slug>/..., NOT at /kaggle/input/<slug>/.
       The older flat layout is what most examples show, and a glob written for
       it silently matches nothing. Hence the leading **, which spans either.
    2. Every kernel output carries a full copy of the repo, so a glob for a bare
       filename finds stale snapshots from earlier runs. Anchoring on the slug
       and asserting exactly one match makes that impossible, not just unlikely.
    """
    hits = glob.glob(f"/kaggle/input/**/{slug}/**/{filename}", recursive=True)
    assert len(hits) == 1, (
        f"expected exactly 1 {filename} under a source named {slug!r}, found {hits}.\\n"
        f"Sources actually mounted: {sorted(glob.glob('/kaggle/input/*/*/*'))}"
    )
    return Path(hits[0])
'''

def prep_cells(smoke: bool = False):
  """Cells for the data-prep session. `smoke` shrinks every input, nothing else.

  The smoke variant runs the identical code path on a few thousand documents, so
  it proves stream -> filter -> tokenizer -> both corpora -> mountable output in
  minutes. The alternative is discovering a broken path two hours into the real
  build, and the real build is the only session in the project nobody has ever
  run before.
  """
  return [
    ("# LiitLLM — data prep SMOKE TEST\n"
     "#\n"
     "# Same code as the real prep, tiny inputs. Proves the whole path works and\n"
     "# that the output is mountable by the training parts. Its output slug is\n"
     "# different, so it can never be mistaken for the real corpora.\n"
     if smoke else
     "# LiitLLM — data prep\n"
     "#\n"
     "# Builds the corpora exactly once. Its kernel output is mounted read-only\n"
     "# by every training part via kernel_sources.\n")
    + "#\n"
    "# **Run this on a CPU session.** It is IO-bound, not GPU-bound, and CPU\n"
    "# sessions do not consume the GPU quota the 20-hour training runs need.\n"
    "# Doing this on a GPU session throws away hours of the budget for nothing.",

    (f"BAKEOFF_N = {2_000 if smoke else 50_000}\n"
     f"TARGET = {2_000_000 if smoke else 200_000_000}      # tokens per corpus\n"
     f"LIMIT = {30_000 if smoke else None}          # max documents read\n"
     f"TOK_SAMPLE = {30_000 if smoke else 50_000}     # documents the BPE trains on\n"
     "SOURCES = (\"fineweb2\", \"hplt\")   # unioned; see data.SOURCES for keep rates"),

    MOUNT,

    FIND_REPO,

    "!pip install -q datasets tokenizers",

    "# Which corpus actually preserved code-switching? Standard pipelines filter\n"
    "# it out, so this measures how much each one threw away — the answer is a\n"
    "# case-study figure either way.\n"
    "from liitllm import data as D\n"
    "D.bakeoff(n=BAKEOFF_N)",

    "# Build both corpora in one pass: same source, same order, same dedup, the\n"
    "# only difference is the Taglish filter. `prepare` asserts they finish at\n"
    "# identical token counts — if they differ, the ablation would be measuring\n"
    "# corpus SIZE rather than corpus content, and that looks exactly like a result.\n"
    "#\n"
    "# Set sources from the bakeoff above.\n"
    "D.prepare(target=TARGET, vocab_size=8192, limit=LIMIT,\n"
    "          tok_sample=TOK_SAMPLE, sources=SOURCES)",

    "# Sanity-check the filter by eye before spending 40 GPU-hours on its output.\n"
    "# A bad lexicon shifts every score without ever raising an error.\n"
    "from liitllm import taglish as tg\n"
    "import itertools\n"
    "for doc in itertools.islice(D._stream(SOURCES[0]), 200):\n"
    "    tl, en = tg.score(doc)\n"
    "    mark = \"KEEP\" if tg.is_taglish(doc) else \"drop\"\n"
    "    print(f\"[{mark} tl={tl:.2f} en={en:.2f}] {doc[:160]}\")",

    "# Stage the corpora at the TOP of /kaggle/working so they become this\n"
    "# kernel's output, which the training parts mount via kernel_sources. No API\n"
    "# credentials and no dataset versioning: the artifacts move as kernel output.\n"
    "#\n"
    "# The repo copy is removed first — every kernel output otherwise carries a\n"
    "# full copy of the repo, which is what makes a bare glob for pyproject.toml\n"
    "# resolve to a stale snapshot in later sessions.\n"
    "import shutil\n"
    "stage = Path('/kaggle/working/data')\n"
    "if stage.exists():\n"
    "    shutil.rmtree(stage)\n"
    "shutil.copytree(REPO / 'data', stage)\n"
    "shutil.rmtree(REPO, ignore_errors=True)\n"
    "for p in sorted(stage.rglob('*')):\n"
    "    if p.is_file():\n"
    "        print(f'{p.relative_to(stage)}  {p.stat().st_size / 1e6:.1f} MB')",
]

# A 20-hour run is split into short numbered parts instead of one long session.
# Short sessions are not just safer against the 12-hour wall — they fail cheaply.
# A bad config or an OOM costs one ~4-hour part, and each part ends with a
# checkpoint pushed off-session, so the next part (on either account) picks up
# where it stopped.
TOTAL_STEPS = 18000
PART_STEPS = 3600      # ~4 hours on a T4; 5 parts per arm
MAX_HOURS = 8.0        # backstop only — stop_at_step should fire first

ARMS = {
    "baseline": "configs/liit-29m.yaml",
    "ablation": "configs/ablation-unfiltered.yaml",
}


def train_cells(arm: str, part: int, n_parts: int):
    config = ARMS[arm]
    stop_at = min(part * PART_STEPS, TOTAL_STEPS) - 1
    return [
        f"# LiitLLM — {arm} training, part {part} of {n_parts}\n"
        "#\n"
        f"# Trains up to step {stop_at}, pushes a checkpoint, and stops. Run part\n"
        f"# {part + 1} next — on either account — and it resumes from here.\n"
        "#\n"
        "# Parts are cut by STEP, not by wall clock, so part boundaries are the\n"
        "# same regardless of how fast the session's GPU happened to be.\n"
        "#\n"
        "# Accelerator must be T4 x2. Every cell below is safe to re-run.",

        f"PART = {part}\n"
        f"CONFIG = '{config}'\n"
        # Part N resumes from part N-1's kernel output. Part 1 points at a slug
        # that cannot exist, so its glob finds nothing and it starts from step 0
        # — which is the correct behaviour for part 1 and an error for any other.
        + (f"PREV_SLUG = '{arm}-part{part - 1}'\n" if part > 1
           else "PREV_SLUG = '__none__'   # part 1 has no predecessor\n")
        + f"STOP_AT_STEP = {stop_at}\n"
        f"MAX_HOURS = {MAX_HOURS}   # backstop if the step estimate is off",

        GPU_ASSERT,

        MOUNT,

        FIND_REPO,

        GATES,

        "# Point the config at the corpora in the prep kernel's output.\n"
        "import yaml\n"
        "cfg_path = REPO / CONFIG\n"
        "cfg = yaml.safe_load(cfg_path.read_text())\n"
        "arm = Path(cfg['data_dir']).name          # 'filtered' or 'unfiltered'\n"
        "data_root = find_one(PREP_SLUG, 'tokenizer.json').parent\n"
        "cfg['data_dir'] = str(data_root / arm)\n"
        "cfg['tokenizer_path'] = str(data_root / 'tokenizer.json')\n"
        "cfg['out_dir'] = '/kaggle/working/out'\n"
        "cfg_path.write_text(yaml.safe_dump(cfg))\n"
        "print(yaml.safe_dump(cfg))",

        "# Restore the previous part's checkpoint from ITS kernel output. Loud\n"
        "# either way: a silent fresh start is the worst failure here, because it\n"
        "# looks exactly like progress for the next four hours.\n"
        "import shutil, torch, glob\n"
        "out = Path('/kaggle/working/out'); out.mkdir(parents=True, exist_ok=True)\n"
        "prev = glob.glob(f'/kaggle/input/**/{PREV_SLUG}/**/ckpt.pt', recursive=True)\n"
        "RESUME = bool(prev)\n"
        "if RESUME:\n"
        "    src = Path(prev[0]).parent\n"
        "    for f in src.iterdir():\n"
        "        if f.is_file():\n"
        "            shutil.copy(f, out)\n"
        "    step = torch.load(out / 'ckpt.pt', weights_only=False)['step']\n"
        "    print(f'=== RESUMING from {src} @ step {step} ===')\n"
        "    assert step < STOP_AT_STEP, (\n"
        "        f'checkpoint is already at step {step}, past this part\\'s target '\n"
        "        f'{STOP_AT_STEP} — you are running an earlier part than you meant to'\n"
        "    )\n"
        + ("else:\n"
           "    print('=== NO CHECKPOINT - starting from step 0 (correct for part 1) ===')"
           if part == 1 else
           "else:\n"
           f"    raise SystemExit(\n"
           f"        'part {part} found no checkpoint. Run part {part - 1} first, or check\\n'\n"
           f"        'that the {arm}-part{part - 1} kernel completed and is attached as a source.\\n'\n"
           f"        'Starting from zero here would silently discard '\n"
           f"        '{(part - 1) * PART_STEPS} steps of training.'\n"
           f"    )"),

        "# Train this part.\n"
        "#\n"
        "# The try/except is the fallback. Whatever happens, /kaggle/working/out\n"
        "# already holds the last atomically-written checkpoint, and it becomes\n"
        "# this kernel's output either way — so an OOM or a CUDA error at hour 3\n"
        "# costs one eval interval, not the part. The repo copy is deleted first so\n"
        "# the output carries only the checkpoint, keeping later globs unambiguous.\n"
        "from liitllm.train import train\n"
        "err = None\n"
        "try:\n"
        "    train(str(cfg_path), resume=RESUME, max_hours=MAX_HOURS,\n"
        "          stop_at_step=STOP_AT_STEP)\n"
        "except Exception as e:\n"
        "    err = e\n"
        "    print(f'TRAINING FAILED: {type(e).__name__}: {e}')\n"
        "    print('The last checkpoint is still in /kaggle/working/out and will be '\n"
        "          'saved as this kernel\\'s output. Rerun this part to continue.')\n"
        "print(sorted(p.name for p in out.iterdir()))",

        "# Where did this part land, and is the run still worth continuing?\n"
        "#   still descending -> run the next part\n"
        "#   flattened        -> stop extending; grow the corpus instead\n"
        "#   turned upward    -> memorising; the best checkpoint is already behind us\n"
        "import csv\n"
        "from liitllm.train import plot_curve\n"
        "plot_curve(out / 'loss.csv')\n"
        "vals = [(int(r['step']), float(r['val_loss']))\n"
        "        for r in csv.DictReader((out / 'loss.csv').open())]\n"
        "best = min(vals, key=lambda v: v[1])\n"
        "print(f'best val {best[1]:.4f} @ step {best[0]}   '\n"
        "      f'last val {vals[-1][1]:.4f} @ step {vals[-1][0]}')\n"
        "print('OVERFITTING - best checkpoint is behind us'\n"
        "      if best[0] < vals[-1][0] else 'still improving - run the next part')",
    ]

EVAL_CELLS = [
    "# LiitLLM — evaluation\n"
    "#\n"
    "# The headline question: does filtering a corpus FOR code-switching produce a\n"
    "# model that code-switches? Both arms are scored with the same scorer that\n"
    "# built the corpora, on identical prompts.",

    GPU_ASSERT,

    MOUNT,

    FIND_REPO,

    "BASELINE = 'liitllm-ckpt-baseline'\n"
    "ABLATION = 'liitllm-ckpt-ablation'",

    "from liitllm.evaluate import compare\n"
    "import glob\n"
    "def ckpt(slug):\n"
    "    hits = glob.glob(f'/kaggle/input/{slug}/**/ckpt.pt', recursive=True)\n"
    "    assert len(hits) == 1, f'expected 1 ckpt.pt under {slug}, found {hits}'\n"
    "    return hits[0]\n"
    "tok = str(find_one(DATA_SLUG, 'tokenizer.json'))\n"
    "compare(ckpt(BASELINE), ckpt(ABLATION), tok, out_dir='/kaggle/working/results')",
]


def notebook(cells):
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src.strip().splitlines(keepends=True)}
            for src in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    n_parts = -(-TOTAL_STEPS // PART_STEPS)  # ceil
    built = [("00_prep_smoke", prep_cells(smoke=True)), ("00_prep", prep_cells())]
    for arm in ARMS:
        for part in range(1, n_parts + 1):
            built.append((f"{arm}_part{part}", train_cells(arm, part, n_parts)))
    built.append(("99_eval", EVAL_CELLS))

    for name, cells in built:
        path = HERE / f"{name}.ipynb"
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.name} ({len(cells)} cells)")

    print(f"\nRun order: 00_prep (CPU) -> "
          f"{' -> '.join(f'baseline_part{i}' for i in range(1, n_parts + 1))} -> "
          f"{' -> '.join(f'ablation_part{i}' for i in range(1, n_parts + 1))} -> 99_eval")
    print(f"{n_parts} parts per arm x {PART_STEPS} steps = {TOTAL_STEPS} steps, "
          f"~4h each. Baseline and ablation are independent — run them on the two "
          f"accounts in parallel.")
