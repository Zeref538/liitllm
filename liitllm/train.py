"""The training loop.

Nothing exotic: sample a batch, predict the next token everywhere at once,
cross-entropy against the true next tokens, AdamW step. The parts that actually
matter for a run finishing successfully are the boring ones —

  warmup      the first few hundred steps use a ramping LR. Adam's variance
              estimates are garbage at step 0; a full-size step then can wreck
              the initialization it never recovers from.
  cosine decay  anneal to ~10% of peak so the model settles instead of
              bouncing around the minimum.
  grad clip   caps the global gradient norm at 1.0. One bad batch producing a
              huge gradient is the usual cause of "loss went to NaN at hour 3".
  checkpoint + resume   see below. Munti's runs were ~6 hours and fit in one
              Kaggle session. These are ~20 hours against a 12-hour wall, so
              resume is not a convenience here, it is the only way a run
              finishes at all.

Resume has to survive four things Munti never had to: a kernel killed mid-write,
a session that ends at an arbitrary step, a resumed run replaying the same
batches, and duplicate rows corrupting the loss curve. Each is handled below and
each is marked, because every one of them silently produces a *plausible* result
rather than an error.

    python -m liitllm.train --config configs/liit-29m.yaml
    python -m liitllm.train --config configs/liit-29m.yaml --resume
    python -m liitllm.train --config configs/liit-29m.yaml --resume --max-hours 11
    python -m liitllm.train --curve out/loss.csv     # write the plot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from . import data as D
from . import tokenizer as tk
from .model import Liit, LiitConfig
from .sample import generate_text

# Fixed prompts, sampled at every eval, so the case study can show what the
# model learned over time from an identical starting point. Taglish seeds: each
# one sets up a code-switch rather than testing plain Tagalog, because
# code-switching is the behaviour this project is actually trying to produce.
PROBE_PROMPTS = [
    "Kahapon nag-",
    "Grabe ang traffic sa",
    "Sorry po, hindi ko alam kung",
]

PARAM_CAP = 35_000_000  # HANDOFF.md hard requirement


def lr_at(step: int, *, lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, split_data, batch_size, block_size, device, iters=50):
    """Average loss over several batches — a single batch is far too noisy to
    tell whether the model actually improved between evals."""
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = D.get_batch(split_data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(payload, ckpt_path: Path):
    """Write a checkpoint without ever destroying the last good one.

    torch.save straight onto ckpt.pt means a kill mid-write leaves a truncated
    file — and it is the only copy, so a 15-hour run becomes unresumable. Write
    to a temp file, then os.replace onto the real name: replace is atomic, so
    ckpt.pt is always either the old checkpoint or the new one, never a partial.
    The previous checkpoint is kept as a second line of defence.
    """
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(payload, tmp)
    if ckpt_path.exists():
        os.replace(ckpt_path, ckpt_path.with_name("ckpt_prev.pt"))
    os.replace(tmp, ckpt_path)


def load_checkpoint(ckpt_path: Path, device):
    """Try the current checkpoint, fall back to the previous one.

    Covers the narrow window where a kill lands between the two os.replace calls
    above, and the wider case of a checkpoint corrupted by a dying filesystem.
    """
    for path in (ckpt_path, ckpt_path.with_name("ckpt_prev.pt")):
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            print(f"loaded checkpoint {path.name} @ step {ck['step']}")
            return ck
        except Exception as e:  # truncated, corrupt, half-written
            print(f"WARNING: {path.name} unreadable ({e}); trying older checkpoint")
    return None


def _truncate_log(log_path: Path, step: int):
    """Drop loss.csv rows after `step`.

    The file is appended to, so a resume from step 8000 after reaching 8500
    would leave 8500's rows in place and then write 8000's again — the curve
    ends up with duplicate and out-of-order steps, which is exactly the artifact
    that goes in the case study. Rewrite it to match what the checkpoint knows.
    """
    if not log_path.exists():
        return
    rows = list(csv.DictReader(log_path.open()))
    keep = [r for r in rows if int(r["step"]) <= step]
    if len(keep) != len(rows):
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
            w.writeheader()
            w.writerows(keep)
        print(f"truncated loss.csv: dropped {len(rows) - len(keep)} rows after step {step}")


def _truncate_samples(samples_path: Path, step: int):
    """Same problem, same fix, for the '## step N (val X)' sections."""
    if not samples_path.exists():
        return
    text = samples_path.read_text(encoding="utf-8")
    out, drop = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("## step "):
            drop = int(line.split()[2]) > step
        if not drop:
            out.append(line)
    samples_path.write_text("".join(out), encoding="utf-8")


def train(config_path: str, resume: bool = False, max_hours: float | None = None,
          stop_at_step: int | None = None):
    """Train, optionally stopping early at a wall-clock or step boundary.

    `stop_at_step` is what splits a 20-hour run into short numbered parts. It is
    preferred over `max_hours` for that job because it is deterministic: part 3
    covers the same steps whether it ran on a fast T4 or a slow one, so the parts
    line up across accounts and reruns. `max_hours` stays as the backstop for
    whatever the step estimate got wrong.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg.get("out_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.get("seed", 1337))  # NFR-3: reproducible
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    bs, block = cfg["batch_size"], cfg["model"]["block_size"]
    total_steps = cfg["max_steps"]
    accum = cfg.get("grad_accum", 1)  # effective batch = batch_size * grad_accum

    # vocab_size is declared in three places that must agree — this config, the
    # tokenizer on disk, and whatever data.prepare() was run with — and nothing
    # else cross-checks them until generation, which is *after* the run. A
    # mismatch either crashes with a CUDA device-side assert or, if the wrong
    # vocab happens to fit, trains a model on garbage. At ~20 GPU-hours a run
    # that is worth catching in the first second.
    assert cfg["model"]["vocab_size"] < 65536, "data.DTYPE is uint16; larger vocab wraps silently"
    # Both ablation arms share one tokenizer, and on Kaggle it lives in a mounted
    # dataset rather than ./data — so it is a config key, not a default path.
    tok_path = cfg.get("tokenizer_path", "data/tokenizer.json")
    try:
        tok_vocab = tk.load(tok_path).get_vocab_size()
        assert tok_vocab == cfg["model"]["vocab_size"], (
            f"tokenizer vocab {tok_vocab} != config vocab {cfg['model']['vocab_size']} — "
            "wrong tokenizer mounted, or data.prepare ran with a different --vocab"
        )
    except FileNotFoundError:
        pass  # synthetic-data tests run without a tokenizer

    # data_dir is what separates the two ablation arms: same architecture, same
    # token budget, different corpus. Everything else in their configs is equal.
    data_dir = cfg.get("data_dir", D.DATA_DIR)
    train_data, val_data = D.load_split("train", data_dir), D.load_split("val", data_dir)
    model = Liit(LiitConfig(**cfg["model"])).to(device)
    n_params = model.num_params()
    assert n_params < PARAM_CAP, f"{n_params:,} params exceeds the {PARAM_CAP:,} cap"
    tokens_seen = total_steps * bs * accum * block
    print(f"device={device} params={n_params:,}")
    print(f"corpus={len(train_data):,} tokens | budget={tokens_seen:,} tokens "
          f"({tokens_seen / max(len(train_data), 1):.1f} epochs)")

    # No weight decay on 1-D params (biases, layernorm gains) — decaying them
    # just shrinks the model's ability to scale activations, for no benefit.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95),
    )

    ckpt_path = out / "ckpt.pt"
    log_path = out / "loss.csv"
    samples_path = out / "samples.md"
    start_step = 0
    ck = load_checkpoint(ckpt_path, device) if resume else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        # Without this the resumed run replays the same random windows the
        # original run already trained on, so hours of compute re-see identical
        # batches. get_batch draws from the global RNG, so restoring it is enough.
        if "rng" in ck:
            torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        # Loud, because a silent fresh start after 10 hours of training is the
        # single worst failure mode this script has.
        print(f"=== RESUMING at step {start_step}/{total_steps} ===")
        _truncate_log(log_path, ck["step"])
        _truncate_samples(samples_path, ck["step"])
    elif resume:
        print("=== no checkpoint found — STARTING FRESH at step 0 ===")

    if not log_path.exists():
        log_path.write_text("step,train_loss,val_loss,lr\n")

    # Mixed precision. The free Kaggle GPU is a T4 (Turing), which has fp16
    # tensor cores but *no* bf16 — so bf16 there silently costs us the speedup.
    # bf16 where available (no scaler needed, its range matches fp32); fp16 plus
    # a gradient scaler otherwise, because fp16's narrow range underflows small
    # gradients to zero without one.
    # Check the compute capability directly, not is_bf16_supported(): that call
    # counts *emulated* bf16 and returns True on cards with no bf16 hardware at
    # all (it said True on a P100), which would pick a path that crawls. Real
    # bf16 starts at Ampere, sm_80.
    amp_dtype = None
    if device.startswith("cuda"):
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    def autocast():
        if amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=amp_dtype)

    print(f"precision: {amp_dtype or 'fp32'}")
    if ck is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])

    def checkpoint_now(step, va):
        save_checkpoint(
            model.checkpoint(
                opt=opt.state_dict(), scaler=scaler.state_dict(), step=step,
                val_loss=va, rng=torch.get_rng_state(),
            ),
            ckpt_path,
        )

    t0 = time.time()
    model.train()
    stopped_early = False
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr=cfg["lr"], warmup=cfg["warmup_steps"], total=total_steps)

        # Gradient accumulation: same effective batch on a card that can't hold
        # it in one go. accum=1 (the default) is the plain single-batch path.
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = D.get_batch(train_data, bs, block, device)
            with autocast():
                _, loss = model(x, y)
            scaler.scale(loss / accum).backward()
        # Unscale before clipping, or we'd be clipping the scaled gradients and
        # the 1.0 threshold would mean nothing.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 100) == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s")

        if (step > 0 and step % cfg["eval_every"] == 0) or step == total_steps - 1:
            with autocast():
                tr = estimate_loss(model, train_data, bs, block, device)
                va = estimate_loss(model, val_data, bs, block, device)
            lr_now = opt.param_groups[0]["lr"]
            print(f"  eval @ {step}: train {tr:.4f} val {va:.4f}")
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([step, f"{tr:.4f}", f"{va:.4f}", f"{lr_now:.2e}"])

            checkpoint_now(step, va)

            # Checkpoint-progression samples (PRD FR-10).
            try:
                tok = tk.load(tok_path)
                with samples_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n## step {step} (val {va:.4f})\n\n")
                    for prompt in PROBE_PROMPTS:
                        text = generate_text(
                            model, tok, prompt, device=device,
                            max_new_tokens=120, temperature=0.8, top_k=200,
                        )
                        f.write(f"> {text}\n\n")
            except FileNotFoundError:
                pass  # no tokenizer yet (e.g. synthetic-data tests) — not fatal

            # Stop on our own terms. Kaggle kills the session at 12 hours with no
            # warning; if that lands mid-checkpoint the run loses up to an entire
            # eval interval. Exiting cleanly just under the wall means the next
            # session resumes from a checkpoint we know is complete.
            # `is not None`, not truthiness: --max-hours 0 and --stop-at-step 0
            # are legitimate values and must not read as "no limit".
            if stop_at_step is not None and step >= stop_at_step:
                print(f"=== reached --stop-at-step {stop_at_step}; stopping cleanly ===")
                print(f"=== next part resumes at step {step + 1}/{total_steps} ===")
                stopped_early = True
                break
            if max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
                print(f"=== hit --max-hours {max_hours} at step {step}; stopping cleanly ===")
                print(f"=== next part resumes at step {step + 1}/{total_steps} ===")
                stopped_early = True
                break

    if not stopped_early:
        print(f"done in {(time.time() - t0) / 60:.1f} min -> {ckpt_path}")
    return stopped_early


def plot_curve(csv_path="out/loss.csv", png_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    # Default the image next to its data, so --curve out-nopos/loss.csv doesn't
    # silently overwrite the main run's plot.
    png_path = png_path or csv_path.with_name("curve.png")
    rows = list(csv.DictReader(csv_path.open()))
    steps = [int(r["step"]) for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [float(r["train_loss"]) for r in rows], label="train")
    plt.plot(steps, [float(r["val_loss"]) for r in rows], label="val")
    plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend()
    plt.title("LiitLLM — training loss"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/liit-29m.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-hours", type=float, default=None,
                   help="save and exit cleanly after this long (Kaggle's wall is 12h)")
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="save and exit at this step — how a long run is split into parts")
    p.add_argument("--curve", nargs="?", const="out/loss.csv", default=None)
    a = p.parse_args()
    if a.curve:
        plot_curve(a.curve)
    else:
        train(a.config, resume=a.resume, max_hours=a.max_hours,
              stop_at_step=a.stop_at_step)
