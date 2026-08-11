# LiitLLM

A 32.8M-parameter transformer written from scratch in PyTorch and trained on
**Taglish** — Tagalog-English code-switched Filipino, the register Filipinos
actually write in online. Free Kaggle GPU, open corpora, ₱0.

Language sibling of [Munti](https://github.com/Zeref538/munti), which proved the same architecture on
English TinyStories. The model code is Munti's, essentially unchanged. The new
work is the data.

**[Read the case study →](https://zeref538.github.io/liitllm/)** · **[Models and results on Kaggle →](https://www.kaggle.com/datasets/johnandreimartinez/liitllm-taglish)**

## Results

Four runs: two arms × two seeds, 60,000 steps each, all trained identically.

| | filtered (Taglish) | unfiltered (control) |
|---|---|---|
| English fraction of generations | **0.206 / 0.164** | 0.061 / 0.070 |
| Taglish share of generations | **0.719 / 0.500** | 0.188 / 0.375 |
| val loss | 2.7445 / 2.7551 | 2.6528 / 2.6352 |

Two numbers per cell: seed 1 / seed 2.

**The filtered models put roughly three times as much English into their output.**
The gap in English fraction is 0.120 against a widest within-arm spread of 0.042 —
a separation of 2.8×, with no overlap between the arms. Filtering a corpus *for*
code-switching produces a model that code-switches.

> **Validation loss is not comparable across arms.** Each model is scored on a
> held-out slice of *its own* corpus, so the two arms sit different exams. Filtered
> Taglish is intrinsically harder to predict — two languages of vocabulary and a
> switch point that can fall anywhere — which is why the control arm's "better"
> 2.635 means nothing. The only apples-to-apples measurement is generation from
> identical prompts, scored by the same scorer.

### Reporting both metrics, including the weaker one

The pre-registered headline was *Taglish share* — the fraction of generations
passing the full `is_taglish` test. It gives a gap of **0.328** against a
within-arm spread of **0.219**: same direction, same conclusion, much thinner
margin. It is a threshold count over 32 samples, so it discards magnitude and
inherits binomial noise. The continuous measure separates cleanly; the
thresholded one barely clears. Both are here because reporting only the
flattering one is how you fool yourself.

## The corpus, measured

Two independently-built web corpora, unioned and deduplicated. Taglish keep rate
measured on **50,000** streamed documents per source:

| source | config | kept by the filter |
|---|---|---|
| FineWeb-2 | `fil_Latn` | **38.0%** |
| HPLT 2.0 | `tgl_Latn` | **39.5%** |

**Filipino web text turns out to be natively code-switched at roughly 38–40%.**
That is far above what "language-ID filters treat code-switching as noise" would
predict, and it partly undercuts this project's own premise — these pipelines did
not throw away nearly as much Taglish as assumed.

It also weakens the ablation's control arm, which is ~40% Taglish rather than
near-zero. The contrast is 100% vs ~40%, not 100% vs nothing, and the honest
framing is: *Filipino web text is already heavily code-switched, and filtering the
rest of the way still measurably changes the model* — which it does, above.

> An earlier version of this table read 46.8% and 34.4%, measured on **2,000**
> documents, and concluded FineWeb-2 preserved more code-switching. At 50,000
> documents the sources are equivalent and the ranking had inverted. A small
> sample does not give you a proportional error; it gives you a confident answer
> that is backwards.

Both corpora were trimmed to **exactly 995,000,000 tokens** so the ablation
measures data quality rather than data quantity. Tokenizer on real Taglish: 8192
vocab, **3.54 chars/token** against Munti's 3.96 on English. Tagalog affix
stacking plus English loanwords compress worse, which is why vocab went
4096 → 8192.

## The idea

Every large web corpus — CC-100, OSCAR, FineWeb-2 — is assembled by a
document-level language-ID filter, and those filters treat code-switching as
low-confidence noise. A Filipino document with heavy English either falls below
the confidence threshold and is dropped, or gets labelled `en` and lands in the
English bucket.

**Standard pipelines filter code-switching out. This one filters for it.**

That inversion is the thesis, and the ablation tests it directly: two models,
same architecture, same token budget, same tokenizer — the only difference is
whether the corpus was filtered for code-switching.

## The ablation

`liitllm/evaluate.py` scores both arms' generations with the same scorer that
built the corpora, and compares the gap between arms against the spread between
seeds within an arm.

**With one seed per arm the verdict is reported as INCONCLUSIVE by design** — a
single-run difference cannot be separated from run-to-run noise, and the code says
so out loud rather than letting it read as a finding. Two seeds per arm is what
makes the result above sayable.

The shared tokenizer is trained on the **unfiltered** corpus, deliberately.
Training it on the filtered corpus would tune the vocabulary to one arm's own
distribution. The two prep runs, executed on different accounts a day apart,
produced byte-identical tokenizers.

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
configs/         liit-33m.yaml + ablation-unfiltered.yaml (+ -seed2 of each)
notebooks/       build_notebooks.py generates the Kaggle notebooks
data/lexicons/   tagalog.txt (431 words), english.txt (647 words)
docs/            the case study site; build_site.py regenerates it from results/
run_pipeline.py  chains Kaggle sessions; survives its own death
finish_run.py    waits for the last run, tops it up if short, then scores
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

- **Hyphens are split.** `nag-commute` matches neither list as one token, and
  neither does `nagcommute` — the affix stripper only finds stems that are
  themselves Tagalog, and `commute` is English. Splitting recovers the English
  half. The `nag-` half still scores nothing, since bare affixes are not lexicon
  entries; that under-counts the Tagalog side of a word already surrounded by
  Tagalog function words, so it costs recall, not correctness.
- **Affixes are stripped on a miss.** Tagalog is agglutinative
  (`trabaho` → `nagtrabaho` → `pinagtrabahuhan`); matching bare roots alone would
  undercount `tl_frac` badly. One prefix and one suffix, plus reduplication.
  Stripping deeper starts inventing matches (`nasa` → `sa`).

Measured separation on the three document types:

```
english  tl=0.00 en=0.87  keep=False
tagalog  tl=0.88 en=0.00  keep=False
taglish  tl=0.54 en=0.27  keep=True
```

## Architecture

| | Munti | LiitLLM |
|---|---|---|
| layers / heads / width | 6 / 6 / 384 | 9 / 8 / 512 |
| block size | 256 | 512 |
| vocab | 4096 | 8192 |
| params | 12.3M | **32,777,728** |
| training tokens | ~330M | **1.97B** |
| GPU-hours | ~6 | **~44** |

`block_size: 512` is not padding the numbers: code-switching is a
*discourse*-level pattern — a writer flips register across clauses — so the
longer context is where the behaviour lives.

`dropout: 0.1`, where Munti used `0.0`. Munti chose 0.0 *because* TinyStories was
large relative to the model (0.6 epochs). Here 32.8M params make ~2 passes over a
995M-token corpus, and the same reasoning inverts.

60,000 steps × 64 × 512 = 1.97B tokens ≈ **60 tokens per parameter**, well past
the Chinchilla-optimal 20 — deliberate, since over-training a small model is
standard practice and buys real quality.

**The models are under-trained anyway.** Validation loss was still descending at
step 60,000 in all four runs; nothing flattened or turned up. The runs stopped
because the weekly GPU quota did, not because training converged. Both arms
stopped at the same step, so the comparison holds; the absolute quality does not
represent a converged model.

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

See [RUNBOOK.md](RUNBOOK.md). One CPU session builds the corpora (~3.3h, no GPU
quota), then **one ~10.9-hour GPU session per run**, four runs across two
accounts at ~22 GPU-h per account per week.

Kaggle kills a session at 12 hours. Rather than splitting a run into parts, each
run is sized to fit one session, with a `--max-hours 11` guard that checkpoints
and exits cleanly before the wall. If a session still ends short — crash, quota,
the guard firing early — `run_pipeline.py` notices the run did not reach its
target step and pushes a continuation that resumes from the last checkpoint. That
path fired twice for real: `ablation` stopped at 59,000 and `ablation2` at 55,000,
and both were topped up to 60,000 without anyone awake.

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
instead of `is not None`. `test_data.py` caught a worse one: deduplicating on a
200-character prefix collapsed a 20,000-token corpus to 109 tokens, because web
pages share navigation boilerplate.

## Can do / can't do

**Can:** produce fluent, register-correct conversational Taglish, switching
between languages mid-sentence the way the corpus does.

> Kahapon nag-iinom ako ng buko juice. Alam mo naman kasi yung feeling na gusto mo
> ng pakulo before ka mag-hard reservation. […] Maybe this time I'm going to be
> better off. Gets ko na lang, and I know I'm just hoping for that.

The control model, same prompt, stays in formal monolingual register — which is
the whole finding:

> Kahapon nag-usap sila ni Jimmy sa telepono. Nang tanungin kung anong oras na ang
> kanilang komunikasyon para sa Skype, tinanong sila ni Jimmy kung ano ang sagot.

**Can't:** facts, arithmetic, code, or instruction following.

> `Ang kabuuang populasyon ng Pilipinas noong 2020 ay` → **umabot sa 3.5 milyon.**

The real figure is about 109 million. The sentence is grammatical, plausibly
shaped, and off by a factor of thirty.

> `1 + 1 =` → **0.04 – 5.05 = 0.48 – 0.8 – 0.8 – 0.2 – 0.8 …**

> `def fibonacci(n):` → **Southern California, Mexico, CA 9003, Mexico, Mexico …**

It also degenerates into repetition at longer horizons (no repetition penalty at
sampling time), invents names and dates fluently, and has no instruction-following
stage — it continues text, it does not answer questions. Its idea of Taglish is
the internet's: blogs, news comments, entertainment copy.

## License

MIT. Corpora are streamed from FineWeb-2 (ODC-By) and HPLT 2.0, not redistributed
here.
