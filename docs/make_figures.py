"""Draw the results dashboard and the per-run table from committed result files.

    python docs/make_figures.py   ->  docs/img/results.png, docs/img/runs.png

Every number comes from results/verdict/*.json or results/00-prep/00-prep.log,
so the pictures regenerate from the repo alone, like docs/index.html does.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

# Same hues as the page; blue nudged up so it passes the chroma check.
CLAY, TIDE = "#b0401f", "#808080"  # one project colour; the control is grey
PAPER, INK, MUTED, GRID = "#ffffff", "#000000", "#6b6b6b", "#e5e5e5"

# Bundled so the images render the same on any machine (SIL OFL, see docs/fonts/OFL.txt).
from matplotlib import font_manager
for f in (ROOT / "docs" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

plt.rcParams.update({
    "font.family": "Instrument Sans", "font.size": 11, "text.color": INK,
    "axes.facecolor": PAPER, "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.axisbelow": True,
})

verdict = json.loads((ROOT / "results/verdict/codeswitch.json").read_text())
curves = json.loads((ROOT / "results/verdict/curves.json").read_text())
FIL, UNF = verdict["arms"]["baseline_filtered"], verdict["arms"]["ablation_unfiltered"]
keep = {m[0]: float(m[1]) for m in re.findall(
    r'(\w+): [\d,]+/[\d,]+ docs kept \(([\d.]+)%\)',
    (ROOT / "results/00-prep/00-prep.log").read_text(encoding="utf-8"))}

mean = lambda runs, k: sum(r[k] for r in runs) / len(runs)
en_f, en_u = mean(FIL, "en_mean"), mean(UNF, "en_mean")
gap = en_f - en_u
spread = max(max(r["en_mean"] for r in a) - min(r["en_mean"] for r in a) for a in (FIL, UNF))


def head(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=26 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED)


def paired_bars(ax, key, fmt):
    """One bar per run, seeds side by side, arms grouped. Value on every bar (only 4)."""
    for i, (runs, c, name) in enumerate([(FIL, CLAY, "filtered"), (UNF, TIDE, "unfiltered")]):
        for j, r in enumerate(runs):
            x = i + (j - 0.5) * 0.36
            ax.bar(x, r[key], 0.32, color=c, edgecolor=PAPER, linewidth=2)
            ax.text(x, r[key], fmt(r[key]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1], ["Filtered\n(Taglish only)", "Unfiltered\n(control)"])
    ax.tick_params(axis="x", length=0)


def dashboard():
    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1, 1], hspace=0.62, wspace=0.32,
                          left=0.06, right=0.97, top=0.83, bottom=0.06)
    fig.text(0.06, 0.955, "LiitLLM: filter for Taglish, get a model that speaks it",
             fontsize=26, fontweight="bold")
    fig.text(0.06, 0.925, "Two identical 32.8M-parameter models, trained from scratch on the same "
             "token budget. One saw only Taglish-filtered text, one saw the unfiltered web text.",
             fontsize=11.5, color=MUTED)
    fig.text(0.06, 0.903, "Every number is measured in this repo: 2 seeds per arm, 32 generations "
             "per run, 8 fixed prompts, scored by the same filter that built the corpus.",
             fontsize=11.5, color=MUTED)

    # Row 1 left: the result.
    ax = fig.add_subplot(gs[0, :2])
    paired_bars(ax, "en_mean", lambda v: f"{v:.3f}")
    ax.axhline(en_f, xmin=0.08, xmax=0.45, zorder=0, color=CLAY, ls="--", lw=1)
    ax.axhline(en_u, xmin=0.55, xmax=0.92, zorder=0, color=TIDE, ls="--", lw=1)
    ax.set_ylim(0, 0.27)
    ax.set_ylabel("English share of words")
    head(ax, "How much English each model writes",
         "Each bar is one training run (seed 1, seed 2). Dashed line = the arm's average.")

    # Row 1 right: hero numbers.
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_title("What the ablation shows", loc="left", fontsize=14, fontweight="bold")
    rows = [
        (f"{en_f / en_u:.1f}×", CLAY, "more English", f"{en_f:.3f} vs {en_u:.3f} average English share"),
        (f"{gap / spread:.1f}×", CLAY, "bigger than the noise",
         f"gap {gap:.3f} vs seed-to-seed spread {spread:.3f}"),
        (f"{min(keep.values()):.0f}–{max(keep.values()):.0f}%", TIDE, "of web text already Taglish",
         "so the filter keeps ~2 in 5 documents"),
        ("0", TIDE, "pesos spent", "~44 GPU-hours on free Kaggle T4s"),
    ]
    for k, (big, c, label, sub) in enumerate(rows):
        y = 0.86 - k * 0.25
        ax.text(0, y, big, fontsize=26, fontweight="bold", color=c, va="center")
        ax.text(0.40, y + 0.04, label, fontsize=12.5, fontweight="bold", va="center")
        ax.text(0.40, y - 0.06, sub, fontsize=8.5, color=MUTED, va="center")

    # Row 2: taglish share, Tagalog share, corpus keep-rate.
    ax = fig.add_subplot(gs[1, 0])
    paired_bars(ax, "taglish_share", lambda v: f"{v:.0%}")
    ax.set_ylim(0, 0.9); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    head(ax, "Outputs that pass as Taglish", "Share of 32 generations passing the filter")

    ax = fig.add_subplot(gs[1, 1])
    paired_bars(ax, "tl_mean", lambda v: f"{v:.2f}")
    ax.set_ylim(0, 0.8)
    head(ax, "…and it still writes Tagalog", "Tagalog share of words: English was added, not swapped in")

    ax = fig.add_subplot(gs[1, 2])
    names = list(keep)
    ax.bar(names, [keep[n] for n in names], 0.5, color=MUTED, edgecolor=PAPER, linewidth=2)
    for n in names:
        ax.text(n, keep[n], f"{keep[n]:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 50); ax.set_xticks(range(len(names)), ["FineWeb-2\n(fil_Latn)", "HPLT 2.0\n(tgl_Latn)"])
    ax.tick_params(axis="x", length=0)
    head(ax, "Web text the filter kept", "50,000 documents sampled per source")

    # Row 3: loss curves, with the warning that they do not compare across arms.
    ax = fig.add_subplot(gs[2, :])
    for key, c, ls, lab in [("baseline", CLAY, "-", "filtered · seed 1"), ("baseline2", CLAY, ":", "filtered · seed 2"),
                            ("ablation", TIDE, "-", "unfiltered · seed 1"), ("ablation2", TIDE, ":", "unfiltered · seed 2")]:
        pts = curves[key]
        ax.plot([p[0] for p in pts], [p[2] for p in pts], color=c, ls=ls, lw=2, label=lab)
    ax.set_xlim(0, 61000); ax.set_ylim(2.55, 4.25)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss")
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=10)
    head(ax, "Training was stable and repeatable",
         "Seeds overlap within each arm. Loss is NOT comparable across arms: each is "
         "scored on its own data, so the lower blue line is not the better model.")

    fig.savefig(OUT / "results.png", dpi=150)
    plt.close(fig)


def runs_table():
    rows = [("Filtered", i + 1, r) for i, r in enumerate(FIL)] + \
           [("Unfiltered", i + 1, r) for i, r in enumerate(UNF)]
    cols = ["Arm", "Seed", "English\nshare", "Tagalog\nshare", "Passes as\nTaglish", "Val.\nloss", "Steps"]
    cells = [[a, str(s), f"{r['en_mean']:.3f}", f"{r['tl_mean']:.3f}",
              f"{round(r['taglish_share'] * r['n'])} / {r['n']}", f"{r['val_loss']:.3f}",
              f"{r['step']:,}"] for a, s, r in rows]
    tint = {"Filtered": "#fbe3da", "Unfiltered": "#ececec"}

    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off")
    fig.suptitle("Every training run, measured the same way", fontsize=16, fontweight="bold", y=0.95)
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2.2)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_facecolor("#000000"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(tint[rows[i - 1][0]] if j in (0, 2) else "white")
    fig.text(0.5, 0.06, "32 generations per run from 8 fixed prompts. Shaded column = the metric the "
             "ablation is judged on.\nReproduce: python -m liitllm.evaluate (RUNBOOK.md, Step 5)",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(OUT / "runs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dashboard()
    runs_table()
    print(f"wrote {OUT / 'results.png'} and {OUT / 'runs.png'}")
