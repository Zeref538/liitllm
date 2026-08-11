# RUNBOOK — the actual sessions

Budget: **~47 GPU-hours** of a 60-hour ceiling. Two Kaggle accounts, ~30 GPU-h
per week each.

| account | role |
|---|---|
| `johnandreimartinez` | main — run the **baseline** arm here |
| `zeref824` | training — run the **ablation** arm here |

Switch the live token by overwriting `~/.kaggle/access_token`. **`printf`, not
`echo`** — a trailing newline breaks the token.

```bash
export PYTHONUTF8=1                       # or the CLI dies writing logs, 0-byte file
printf '<token>' > ~/.kaggle/access_token
python -m kaggle config view | grep username   # confirm which is live
```

## Before anything

```bash
python test_liitllm.py && python test_resume.py && python test_data.py
python notebooks/build_notebooks.py
```

All four gates are CPU and take under two minutes. They are the cheapest thing in
this project and they gate the most expensive.

## The runner

`run_pipeline.py` chains the sessions: it pushes a kernel, polls until it reaches
a terminal state, pulls the output, and pushes the next one. **On any failure it
stops the chain, downloads the kernel log, and prints the traceback** — nothing
downstream is pushed, because continuing past a failed part would train on a
checkpoint that never advanced.

```bash
python run_pipeline.py --arm repo            # publish this repo as a dataset
python run_pipeline.py --arm prep-smoke      # ~10 min end-to-end proof
python run_pipeline.py --arm prep            # the real corpus build
python run_pipeline.py --arm baseline        # parts 1..5, chained
python run_pipeline.py --arm baseline --parts 3 4 5   # resume mid-chain
```

Artifacts move between sessions as **kernel output**, mounted by the next session
via `kernel_sources` — not as datasets versioned from inside the notebook. That
needs no API credentials in Kaggle Secrets and has nothing that can fail at the
end of a 4-hour run.

Datasets are **private and per-account**, so each account uploads its own repo
copy and builds its own corpora. Publishing one shared public copy would be less
work, but it would publish your code as a side effect of a plumbing decision.
`--arm prep` is a CPU session, so the duplication costs wall-clock and zero GPU
quota. Pass `--public` if you'd rather share one copy.

## Step 0 — upload the repo

```bash
python run_pipeline.py --arm repo
```

`-r zip` on **every** call, create and version alike — without it the CLI silently
drops every subdirectory and reports success. The runner does this and verifies
with `datasets files` afterwards.

**Re-run this after any change to `liitllm/`.** The kernels copy the package out
of the mounted dataset, so an un-uploaded edit does not exist as far as Kaggle is
concerned — the session runs the previous version and looks completely fine.

The runner then waits for `datasets status` to report `ready` before pushing any
kernel. Uploading returns as soon as the bytes transfer, but Kaggle keeps
processing afterwards and **mounts the dataset empty in the meantime** — a kernel
pushed into that window dies with `expected exactly 1 pyproject.toml ... found []`,
which reads like a missing file and is really a race. (This is not hypothetical;
it took out the first prep run.)

## Step 1 — build the corpora (CPU session, 0 GPU-hours)

Prove the path first — it takes ten minutes and the real build is the only
session in this project nobody has ever run:

```bash
python run_pipeline.py --arm prep-smoke   # same code, ~30k docs, 2M tokens
```

Then the real thing:

```bash
python run_pipeline.py --arm prep
```

Both run on a **CPU** session. Streaming and filtering 700M tokens is IO-bound;
running it on a GPU session would burn hours of the quota the training runs need,
for nothing.

It runs the corpus bake-off, builds both corpora in one pass, and publishes
`liitllm-data`. **Read the filter samples it prints before moving on** — a bad
lexicon shifts every score without raising an error, and it decides the training
corpus, the ablation, *and* the headline metric at once.

Make `liitllm-data` and both checkpoint datasets **public** so the second account
can mount them without sharing credentials.

Then update `max_steps` in both configs from the token count it reports:
`max_steps ≈ 20 × params / (batch_size × block_size)`. `train.py` prints the
resulting epoch count on startup — check it before letting a part run.

## Step 2 — train, five parts per arm

Run the two arms **in parallel on the two accounts**. They are independent, so
this halves wall-clock and keeps each account inside its weekly quota.

| account | notebooks |
|---|---|
| main | `baseline_part1` → … → `baseline_part5` |
| training | `ablation_part1` → … → `ablation_part5` |

Each part: ~3,600 steps, ~4 hours, pushes its checkpoint, stops. Attach as data
sources: `liitllm-repo`, `liitllm-data`, and the arm's checkpoint dataset.

```bash
python -m kaggle kernels push -p notebooks --accelerator NvidiaTeslaT4
python -m kaggle kernels status <user>/<slug>
```

**Pin the accelerator.** The default is a P100 (sm_60) that Kaggle's own PyTorch
does not support, and everything before the first training step is CPU work — so
a doomed session looks healthy for ten minutes. The first cell asserts sm_70+.

### If a part fails

The training cell catches the exception, still pushes the checkpoint, then
re-raises. So:

- **Crashed mid-part** — rerun the same part. It resumes from the last eval.
- **Push failed 3×** — do *not* close the session. Save a notebook version so
  `/kaggle/working` survives as kernel output, then mount that output next time.
- **Ran the wrong part** — the resume guard catches it: a part whose checkpoint
  is already past its target refuses to run, and a part >1 with no checkpoint
  refuses to start from zero.
- **OOM at `block_size: 512`** — set `batch_size: 32` and `grad_accum: 2` in both
  configs. Same effective batch. Change *both* or the ablation is confounded.

## Step 3 — the week-1 gate

The last cell of each part prints best-vs-last val loss. After part 5, read it:

| val curve | meaning | next |
|---|---|---|
| still descending | still learning | extend — the quota is there |
| flattened | corpus exhausted of what it can teach | stop extending; grow the corpus on CPU sessions |
| turned upward | memorising | roll back to the best checkpoint; more training makes it worse |

## Step 4 — second seed

Change `seed:` in both configs, push to `*-seed2` checkpoint slugs, rerun.

**This is not optional if you want to claim the filter worked.** With one seed
per arm, `evaluate.py` reports INCONCLUSIVE by design — a single-run gap cannot
be separated from run-to-run noise.

## Step 5 — evaluate

Run `notebooks/99_eval.ipynb` with all checkpoints attached. It writes
`results/codeswitch.json` and per-arm generation dumps including the failure
probes.

Report the ablation as a **range across seeds**. If the gap between arms does not
exceed the spread within an arm, the honest conclusion is "no measurable effect"
— write that up. Do not move the scorer's thresholds afterwards: they decided the
training corpus, so retuning them to improve the result makes the metric
circular.

## Afterwards

Rotate both API tokens — Kaggle → Settings → API → Expire API Token.
