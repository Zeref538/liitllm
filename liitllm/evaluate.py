"""Does filtering a corpus FOR code-switching produce a model that code-switches?

That is the whole experiment, and this module answers it with one number: the
English fraction of each model's generations, scored by the same scorer that
built the corpora.

    python -m liitllm.evaluate --baseline out/ckpt.pt --ablation out-unfiltered/ckpt.pt

Reading the result honestly
---------------------------
The comparison is only meaningful against run-to-run noise. Pass several
checkpoints per arm (one per seed) and the report prints each arm's spread
alongside the gap between arms. **If the gap between arms does not exceed the
spread within an arm, the honest conclusion is "no measurable effect"** — and
that is a publishable finding, not a failed experiment. Do not tune the
thresholds until the answer looks better; the scorer's thresholds decided the
training corpus, so moving them afterwards makes the metric circular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import taglish as tg
from . import tokenizer as tk
from .model import Liit
from .sample import generate_text

# Held-out Taglish seeds. Fixed and committed so every checkpoint, every seed and
# every rerun is compared on identical inputs.
EVAL_PROMPTS = [
    "Kahapon nag-",
    "Grabe ang traffic sa",
    "Sorry po, hindi ko alam kung",
    "Nag-order ako ng",
    "Ang ganda ng",
    "Pagod na pagod ako kasi",
    "Tapos sabi niya sa akin na",
    "Bukas may meeting kami tungkol sa",
]

# Deliberately outside anything the corpus contains. Munti's case study earned
# its credibility from printing these rather than hiding them.
FAILURE_PROMPTS = [
    "The mitochondria is the",
    "def fibonacci(n):",
    "1 + 1 =",
    "Ang kabuuang populasyon ng Pilipinas noong 2020 ay",
]


def _load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    return Liit.from_checkpoint(ck, device=device), ck.get("step"), ck.get("val_loss")


def generations(model, tok, prompts, device="cpu", n_per_prompt=4, seed=1234, **kw):
    """Sample from each prompt. Fixed seed so two arms see identical randomness."""
    torch.manual_seed(seed)
    out = []
    for prompt in prompts:
        for _ in range(n_per_prompt):
            out.append((prompt, generate_text(model, tok, prompt, device=device, **kw)))
    return out


def codeswitch_rate(texts) -> dict:
    """Summarise the English fraction across a set of generations."""
    en = sorted(tg.score(t)[1] for t in texts)
    tl = sorted(tg.score(t)[0] for t in texts)
    n = len(en)
    mid = n // 2
    return {
        "n": n,
        "en_mean": sum(en) / max(n, 1),
        "en_median": en[mid] if n else 0.0,
        "tl_mean": sum(tl) / max(n, 1),
        # Share of generations that would themselves pass the corpus filter — the
        # most direct reading of "does this model produce Taglish?"
        "taglish_share": sum(1 for t in texts if tg.is_taglish(t)) / max(n, 1),
    }


def _arm(ckpts, tok, device, label, out_dir: Path):
    """Score one arm, one entry per seed."""
    seeds = []
    for ckpt in ckpts:
        model, step, val = _load(ckpt, device)
        gens = generations(model, tok, EVAL_PROMPTS, device=device,
                           max_new_tokens=120, temperature=0.8, top_k=200)
        stats = codeswitch_rate([g for _, g in gens])
        stats.update(ckpt=str(ckpt), step=step, val_loss=val)
        seeds.append(stats)

        with (out_dir / f"{label}_generations.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {label} — {Path(ckpt).parent.name} (step {step}, val {val})\n\n")
            for prompt, text in gens:
                t_tl, t_en = tg.score(text)
                f.write(f"**`{prompt}`** — tl={t_tl:.2f} en={t_en:.2f}\n\n> {text}\n\n")
            f.write(f"\n### failure probes (out of domain)\n\n")
            for prompt, text in generations(model, tok, FAILURE_PROMPTS, device=device,
                                            n_per_prompt=1, max_new_tokens=100,
                                            temperature=0.8, top_k=200):
                f.write(f"**`{prompt}`**\n\n> {text}\n\n")
    return seeds


def compare(baseline, ablation, tokenizer_path="data/tokenizer.json",
            out_dir="results", device=None):
    """Score both arms and write the verdict. Accepts one checkpoint or several."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    baseline = [baseline] if isinstance(baseline, (str, Path)) else list(baseline)
    ablation = [ablation] if isinstance(ablation, (str, Path)) else list(ablation)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = tk.load(tokenizer_path)

    arms = {
        "baseline_filtered": _arm(baseline, tok, device, "baseline_filtered", out_dir),
        "ablation_unfiltered": _arm(ablation, tok, device, "ablation_unfiltered", out_dir),
    }

    def means(seeds):
        return [s["taglish_share"] for s in seeds]

    b, a = means(arms["baseline_filtered"]), means(arms["ablation_unfiltered"])
    gap = sum(b) / len(b) - sum(a) / len(a)
    # Widest within-arm spread. With one seed per arm this is 0 and the gap
    # cannot be distinguished from noise — which the verdict says out loud
    # rather than letting a single-run difference read as a result.
    spread = max(max(b) - min(b), max(a) - min(a))
    conclusive = len(b) > 1 and len(a) > 1 and abs(gap) > spread

    verdict = {
        "arms": arms,
        "taglish_share_gap": gap,
        "within_arm_spread": spread,
        "seeds_per_arm": [len(b), len(a)],
        "conclusive": conclusive,
        "reading": (
            "filtering increased code-switching beyond run-to-run noise" if conclusive and gap > 0
            else "filtering DECREASED code-switching beyond noise" if conclusive
            else "no measurable effect: the gap does not exceed within-arm spread"
            if len(b) > 1 and len(a) > 1
            else "INCONCLUSIVE: one seed per arm cannot separate a real gap from noise — "
                 "run a second seed before claiming anything"
        ),
    }
    (out_dir / "codeswitch.json").write_text(json.dumps(verdict, indent=2))

    print(f"\n{'arm':<24} {'taglish share':>14} {'en_mean':>9} {'val':>8}")
    for name, seeds in arms.items():
        for s in seeds:
            val = f"{s['val_loss']:.4f}" if s["val_loss"] is not None else "-"
            print(f"{name:<24} {s['taglish_share']:>14.3f} {s['en_mean']:>9.3f} {val:>8}")
    print(f"\ngap {gap:+.3f}   within-arm spread {spread:.3f}")
    print(f"VERDICT: {verdict['reading']}")
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", nargs="+", required=True, help="one ckpt.pt per seed")
    p.add_argument("--ablation", nargs="+", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    compare(a.baseline, a.ablation, a.tokenizer, a.out)
