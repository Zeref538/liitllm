# LiitLLM

A ~29M-parameter transformer written from scratch in PyTorch and trained on
**Taglish** — Tagalog-English code-switched Filipino, the register Filipinos
actually write in online. Free Kaggle GPU, open corpora, ₱0.

Language sibling of [Munti](../Munti), which proved the same architecture on
English TinyStories. The model code is Munti's, essentially unchanged. The new
work is the data.

> **Status: built, not yet trained.** Every gate passes and the pipeline runs end
> to end on synthetic data. The results sections below are empty on purpose —
> they get filled in from the actual runs, not from expectations.

## The idea

Every large web corpus — CC-100, OSCAR, FineWeb-2 — is assembled by a
document-level language-ID filter, and those filters treat code-switching as
low-confidence noise. A Filipino document with heavy English either falls below
the confidence threshold and is dropped, or gets labelled `en` and lands in the
English bucket.

**Standard pipelines filter code-switching out. This one filters for it.**

That inversion is the thesis, and the ablation tests it directly: two models,
same architecture, same seed, same token budget, same tokenizer — the only
difference is whether the corpus was filtered for code-switching.

## Results

*Pending the first run. `results/` will hold the loss curves, checkpoint
progression samples, and `codeswitch.json`.*

| | filtered | unfiltered |
|---|---|---|
| val loss | — | — |
| Taglish share of generations | — | — |

## The ablation

The question is not "does the model work" but "did the filter cause anything".
`liitllm/evaluate.py` scores both arms' generations with the same scorer that
built the corpora, and compares the gap between arms against the spread between
seeds within an arm.

**With one seed per arm the verdict is reported as INCONCLUSIVE by design.** A
single-run difference cannot be separated from run-to-run noise, and the code
says so out loud rather than letting it read as a finding.

## What's here

```
liitllm/
  model.py       the transformer — unchanged from Munti, language-agnostic
  tokenizer.py   byte-level BPE, trained here on our own corpus
  taglish.py     the code-switch scorer — the centrepiece
  data.py        streamed web text -> filtered + unfiltered corpora
  train.py       training loop, crash-safe checkpointing, part-wise stopping
  sample.py      generation
  evaluate.py    the ablation verdict
configs/         liit-29m.yaml (baseline) + ablation-unfiltered.yaml
notebooks/       build_notebooks.py generates the Kaggle notebooks
data/lexicons/   tagalog.txt, english.txt
```

## The Taglish scorer

Two lexicons, two fractions. `tl_frac` is the share of word tokens matching a
Tagalog list weighted toward **function words** — a closed class a writer cannot
produce Tagalog syntax without. `en_frac` matches a high-frequency English list
that deliberately includes **content words**, because the signature form of
Taglish borrows English content into Tagalog syntax:

> "nag-commute ako sa office, na-late pa rin sa meeting"

where the English contribution is almost entirely nouns and verbs. Scoring
English on function words alone would rate that near 0% English and discard
exactly the documents the project is looking for.

Two details that matter more than they look:

- **Hyphens are split.** Tagalog affixes attach to English roots across a hyphen
  (`nag-commute`, `na-cancel`), so splitting yields a Tagalog affix and an
  English root and the construction scores on both sides instead of neither.
- **Affixes are stripped on a miss.** Tagalog is agglutinative; matching bare
  roots alone would undercount `tl_frac` badly.

Measured separation on the three document types:

```
english  tl=0.00 en=0.87  keep=False
tagalog  tl=0.88 en=0.00  keep=False
taglish  tl=0.54 en=0.27  keep=True
```

## Architecture

| | Munti | LiitLLM |
|---|---|---|
| layers / heads / width | 6 / 6 / 384 | 8 / 8 / 512 |
| block size | 256 | 512 |
| vocab | 4096 | 8192 |
| params | 12.3M | **29,630,976** |

`block_size: 512` is not padding the numbers: code-switching is a
*discourse*-level pattern — a writer flips register across clauses — so the
longer context is where the behaviour lives.

`dropout: 0.1`, where Munti used `0.0`. Munti chose 0.0 *because* TinyStories was
large relative to the model (0.6 epochs). Here ~29M params train ~3 passes over a
~200M-token corpus, and the same reasoning inverts.

18,000 steps × 64 × 512 = 590M tokens ≈ **20.3 tokens per parameter**, from the
Chinchilla rule rather than a round number.

## Run it

```bash
pip install -e .
python test_liitllm.py          # transformer + training loop correctness
python test_resume.py           # a killed session can resume
python test_data.py             # the two corpora are a fair comparison
python -m liitllm.taglish demo  # the scorer's bands
```

All four are CPU, seconds to a minute, and need no dataset or tokenizer.

## The full run (free Kaggle, two accounts)

See [RUNBOOK.md](RUNBOOK.md). Short version: one CPU session builds the corpora
once, then five ~4-hour GPU parts per arm, with the two arms running in parallel
on the two accounts.

Training is **split into numbered parts** rather than one long session. Kaggle
kills a session at 12 hours, and a 20-hour run cannot fit in one. Each part cuts
at a fixed *step* (not a wall-clock time, so parts line up across machines of
different speeds), pushes its checkpoint to a Kaggle Dataset, and stops. The next
part — on either account — resumes from it.

## The correctness gate

`test_liitllm.py` is inherited from Munti unchanged. It uses synthetic token ids,
so it transferred for free: init loss ≈ ln(vocab), no future leakage through the
causal mask, fused attention matching the reference implementation, `get_batch`
targets shifted by exactly one, and an overfit gate that drives 4 batches to
~0 loss.

It is the reason any bad Taglish output is attributable to the data rather than a
bug, which is the entire premise of reusing a proven codebase.

`test_resume.py` is new, and earned its keep immediately — it caught a real bug
where `--max-hours 0` was read as "no limit" because the guard tested truthiness
instead of `is not None`.

## Can do / can't do

*Pending real generations. This section gets written from `results/`, including
the embarrassing ones.*

## License

MIT.
