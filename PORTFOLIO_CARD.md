# LiitLLM — portfolio integration brief

*Paste this whole file into a session working in the `Portfolio` repo. It is
self-contained: what the project is, the verified numbers, and the exact edit.*

## What LiitLLM is

A **32.8M-parameter language model built from scratch** in pure PyTorch, trained
on **Taglish** — Tagalog-English code-switched Filipino, the register Filipinos
actually write in online. The thesis is an inversion: every large web corpus is
sorted by a language-ID filter that treats code-switching as low-confidence noise
and drops it, so this project filters *for* code-switching instead of against it,
and then runs a controlled ablation to prove the filter is what caused the
behaviour.

It is the second **from-scratch** project alongside [Munti](../Munti), and the
harder one: Munti proved the architecture on English, LiitLLM is the same
architecture pointed at a data problem with a real experimental result.

| fact | value |
|---|---|
| params | 32,777,728 (9 layers, 8 heads, 512 wide, 512 context) |
| tokenizer | byte-level BPE, 8192 vocab, 3.54 chars/token, trained on the corpus |
| corpus | FineWeb-2 `fil_Latn` + HPLT 2.0 `tgl_Latn`, streamed → **995,000,000 tokens per arm**, trimmed to exact parity |
| training | 4 runs × 60,000 steps × 64 × 512 (~2 epochs), ~10.9h each, **~44 GPU-hours**, two free Kaggle accounts, ₱0 |
| result | filtered models emit **3× the English** of controls: 0.185 vs 0.065 English fraction, gap **2.8×** the within-arm spread, no overlap |
| ablation | 2 arms × 2 seeds, identical token budget and tokenizer; the only difference is the code-switch filter |
| honesty | reports the weaker pre-registered metric too (0.328 gap vs 0.219 spread), states that cross-arm val loss is not comparable, and states the models are under-trained |

The four things that make it portfolio-worthy, in order:

1. **A measurement that undercut the project's own premise, published anyway.**
   The pitch was "corpora filter code-switching out." Measured on 50,000
   documents per source, Filipino web text is *already* 38–40% code-switched. That
   weakens the control arm and shrinks the claim, and it is stated in the README,
   the case study and the site rather than quietly dropped.
2. **A two-seed ablation with a stated noise floor.** The verdict compares the
   between-arm gap against the within-arm spread, and the evaluator is written to
   return INCONCLUSIVE when given one seed per arm — the honesty is enforced in
   code, not promised in prose.
3. **Two metrics reported, including the one that barely clears.** The continuous
   measure separates cleanly (2.8× the spread); the pre-registered thresholded one
   is much thinner (0.328 vs 0.219). Both are published, with the reason the
   thresholded one is noisier.
4. **An 11-hour training run on a platform that kills sessions at 12.** Atomic
   checkpoints, saved RNG state, a wall-clock guard, a self-resuming runner, and a
   lockfile — every one added after the corresponding failure actually happened.
   The final run stopped at step 55,000 and the automation topped it up to 60,000
   unattended.

**Links:** repo `https://github.com/Zeref538/liitllm` *(create — not pushed yet)* ·
live case study `https://zeref538.github.io/liitllm/` *(publish `docs/`)* ·
Kaggle notebooks below.

### Kaggle links — all on the main account

Profile: https://www.kaggle.com/johnandreimartinez

**All four models, weights and results, one public dataset:**
https://www.kaggle.com/datasets/johnandreimartinez/liitllm-taglish

Contains `{filtered,unfiltered}-seed{1,2}/` — each with a weights-only checkpoint
(131MB, optimizer state stripped), its loss curve and its generations — plus the
verdict JSON, the shared tokenizer, the lexicons, and both writeups.

| run | notebook |
|---|---|
| corpus build | https://www.kaggle.com/code/johnandreimartinez/00-prep |
| filtered, seed 1 | https://www.kaggle.com/code/johnandreimartinez/baseline-part1 |
| unfiltered, seed 1 | https://www.kaggle.com/code/johnandreimartinez/ablation-part1 |
| unfiltered, seed 1 (top-up) | https://www.kaggle.com/code/johnandreimartinez/ablation-part2 |

**Notebooks are private until set to Public in the Kaggle UI** (each notebook →
Settings → Visibility → Public). The CLI cannot flip visibility without re-running
the notebook, which for a training notebook means a failed or cancelled latest
version whose Output tab is empty — destroying the evidence the page exists to
show.

**On the seed-2 runs.** They executed on a second account used purely for GPU
quota, and a Kaggle kernel cannot be transferred between accounts. Re-running them
on the main account would cost ~22 GPU-hours to regenerate artifacts that already
exist, so their *outputs* are in the dataset above instead and the second account
stays private. The quota split is an operational detail, not part of the result;
all four runs' curves and checkpoints are published together.

---

## What to paste where

The block below is a JavaScript object; it goes into `Portfolio/src/data.js`, in
the `projects` array, next to the Munti card (currently at line 51).

**No `App.jsx` change is needed.** `projGroups` (App.jsx:235) already contains
`"Building LLMs"`, which is the group Munti uses and the correct one here.

**Still needed:** the `images` paths don't exist yet — screenshot the case study
site and save as `Portfolio/public/projects/liitllm-*.jpg`, or trim the array to
what you actually have.

```js
  {
    title: "LiitLLM — A Taglish LLM From Scratch",
    groups: ["Building LLMs"],
    description:
      "Built a 32.8M-parameter language model from scratch in pure PyTorch and trained it on Taglish — the Tagalog-English code-switched register Filipinos actually write in — on a thesis that inverts standard practice: every large web corpus is sorted by a language-ID filter that scores code-switched text as low-confidence noise and discards it, so I wrote a filter that keeps only that text. The result is a controlled experiment rather than a demo: two arms, two seeds each, identical architecture, identical 995,000,000-token budget and a shared tokenizer trained on the unfiltered side so neither arm gets an advantage, with the code-switch filter as the only difference. The filtered models emit three times as much English as the controls — a gap 2.8x the widest spread between seeds of the same arm, with no overlap. The finding I did not want is in the writeup too: measured on 50,000 documents per source, Filipino web text is already 38-40% code-switched, which weakens my own control arm and shrinks the claim from 'filtering creates code-switching' to 'filtering the rest of the way still measurably changes the model'.",
    tags: ["PyTorch", "Transformers", "From Scratch", "Code-Switching", "NLP", "Kaggle", "Python"],
    metric: "32.8M params from scratch · 3x code-switching vs control",
    category: "From Scratch · Transformers · Ablation Design",
    date: "2026",
    image: "/projects/liitllm-1.jpg",
    images: [
      "/projects/liitllm-1.jpg",
      "/projects/liitllm-2.jpg",
      "/projects/liitllm-3.jpg",
      "/projects/liitllm-4.jpg",
    ],
    link: "https://github.com/Zeref538/liitllm",
    demo: "https://zeref538.github.io/liitllm/",
    demoLabel: "case study",
    highlights: [
      "Designed the ablation so it could fail: two seeds per arm, exactly equal 995,000,000-token budgets, and one shared tokenizer trained on the unfiltered corpus so the filtered arm gains no vocabulary advantage — the evaluator returns INCONCLUSIVE by construction when given a single seed per arm",
      "Measured 38-40% native code-switching in Filipino web text on 50,000 documents per source, which undercut my own premise and weakened the control arm, and published it as the finding rather than dropping it — an earlier 2,000-document run had given the opposite ranking, so the sample size is now stated next to the number",
      "Reported both metrics including the weaker one: the continuous English-fraction measure separates the arms at 2.8x the within-arm spread with no overlap, while the pre-registered thresholded metric gives 0.328 against a 0.219 spread, and the writeup explains that a threshold count over 32 samples discards magnitude and inherits binomial noise",
      "Trained four ~11-hour runs on a platform that kills sessions at 12 hours, with atomic checkpoint writes, saved RNG state, a wall-clock guard and a self-resuming runner — the last run stopped at step 55,000 and the automation detected it, pushed a continuation from the last checkpoint and finished to 60,000 unattended",
    ],
  },
```

## Suggested screenshots

| file | what |
|---|---|
| `liitllm-1.jpg` | the case study hero — the live code-switch scorer with a Taglish sentence tinted by lexicon |
| `liitllm-2.jpg` | the ablation gap chart (English fraction, both arms, both seeds) |
| `liitllm-3.jpg` | the filtered-vs-unfiltered generation pair on the same prompt |
| `liitllm-4.jpg` | the four validation-loss curves, both seeds overlapping per arm |

Take 1, 2 and 4 from `docs/index.html`; 3 is in
`results/verdict/*_generations.md`.
