# LiitLLM — teaching a small model to code-switch

> A 32.8M-parameter transformer, written in pure PyTorch, trained from scratch on
> **Taglish** — Tagalog-English code-switched Filipino. Four runs, 60,000 steps
> each, ~44 GPU-hours on free Kaggle T4s. The filtered models produce **three
> times** as much English as the controls. Total cost: ₱0.

Every large web corpus is sorted by a language-ID filter, and those filters treat
code-switching as noise. So the register most Filipinos actually write in is
thinned out of the data before any model sees it.

This project inverts that one step: instead of filtering code-switching out, it
filters *for* it. Here is what happened, including the measurement that undercut
my own premise and the bug I shipped into two 11-hour runs.

---

## 1. How it works

A language model does one job: given some tokens, put a probability on every
possible next token. Train it on text where writers flip between Tagalog and
English mid-sentence, and the switch itself becomes something it can predict.

The architecture is [Munti](https://github.com/Zeref538/munti)'s, essentially unchanged — a decoder-only
transformer, 9 layers, 8 heads, 512-wide, weight-tied embeddings. That reuse is
deliberate and it is load-bearing: because the model code was already proven
correct on English, **any difference in Taglish output is attributable to the
data rather than to a bug in the attention mask.** The correctness gate
(`test_liitllm.py`) transferred for free, since it runs on synthetic token ids
and never touches a real corpus.

Two things changed for this subject:

- **`block_size: 512`**, double Munti's. Code-switching is a *discourse*-level
  behaviour — a writer flips register across clauses and sentences, not inside a
  three-word window. At 256 tokens the model would rarely see a full switch cycle.
- **`dropout: 0.1`**, where Munti used `0.0`. Munti used 0.0 *because* TinyStories
  was large relative to the model — 0.6 epochs, nothing to memorise. Here the run
  makes ~2 passes over the corpus, and the same reasoning inverts.

## 2. The decision that made the project: scoring for code-switching

The centrepiece is `liitllm/taglish.py`, about thirty lines, no dependencies. It
splits text into words and asks how many appear in a Tagalog list and how many in
an English list. Documents where **both** fractions clear a floor are kept.

The same function decides the training corpus, the ablation, *and* the headline
metric. That triple duty is why it had to be simple enough to explain in a
paragraph and simple enough to audit by eye.

**The two lists are built on opposite principles, and that asymmetry is the whole
trick.** The Tagalog list is almost entirely function words — `ang`, `ng`, `sa`,
`kasi`, `naman`. Those are a closed class: you cannot produce Tagalog syntax
without them, and English text will not emit them by accident. The English list
deliberately includes *content* words — `office`, `traffic`, `meeting` — because
the signature form of Taglish is English content borrowed into Tagalog syntax:

> nag-commute ako sa office

There is not one English function word in that sentence. Score English the way I
scored Tagalog and the most characteristically Taglish sentence in the corpus
reads as pure Tagalog.

Two mechanics matter more than they look:

**Hyphens get split.** `nag-commute` matches neither list as a single token — and
neither does `nagcommute`, because the affix stripper only finds stems that are
themselves in the Tagalog list, and `commute` is English. Splitting recovers the
English half. The `nag-` half still scores nothing, since bare affixes are not
lexicon entries. That is a real limitation: it under-counts the Tagalog side of a
word that is already surrounded by Tagalog function words, so it costs recall
rather than correctness.

**Affixes get stripped.** Tagalog is agglutinative — `trabaho`, `nagtrabaho`,
`trabahuhan`, `pinagtrabahuhan`. A bare root list misses all of those. The scorer
peels one prefix and one suffix and handles reduplication (`nagcocommute` →
`commute`). It stops at one layer on purpose: two layers reduces `nasa` to `sa`
and starts inventing matches.

## 3. The measurement that undercut my own premise

My pitch was that code-switching gets filtered out of web corpora. Before
building anything on that, I measured it.

| source | config | kept by the filter |
|---|---|---|
| FineWeb-2 | `fil_Latn` | **38.0%** |
| HPLT 2.0 | `tgl_Latn` | **39.5%** |

**Filipino web text is already ~40% code-switched**, straight out of the corpus,
untouched. The langid filter thins it but comes nowhere near removing it.

That is a problem for my framing and a bigger problem for my control arm: the
"unfiltered" model does not train on code-switch-free text, it trains on text
that is already 40% Taglish. The contrast is 100% vs 40%, not 100% vs nothing.

I kept the design and changed the claim. The honest version is not "filtering
creates code-switching from nothing" but **"Filipino web text is already heavily
code-switched, and filtering the rest of the way still measurably changes the
model."** That is a smaller result than I set out to prove, and it is the one the
data supports.

### The version of this table I nearly shipped

I first ran that bake-off on **2,000** documents: FineWeb-2 46.8%, HPLT 34.4%. I
wrote it up as a clear win for FineWeb-2 and moved on. At 50,000 documents the two
sources are equivalent and **the ranking had inverted.**

A small sample does not hand you a proportional error you can reason about. It
hands you a confident answer that is backwards. Sample sizes now get stated next
to every number I report, or the number gets re-run before anyone concludes from
it.

## 4. Tokenizer

Byte-level BPE, 8192 vocab, trained here on this corpus — nothing downloaded.
Measured **3.54 characters per token**, against Munti's 3.96 on English. Tagalog's
affix stacking plus English loanwords compress worse, which is exactly why the
vocabulary went 4096 → 8192.

**It is trained on the *unfiltered* corpus, deliberately, and that has to be
disclosed.** Training the tokenizer on the filtered corpus would tune the
vocabulary to the filtered model's own distribution and hand that arm an advantage
that has nothing to do with the experiment. Neutral by construction.

The two data-prep runs ran on two different accounts a day apart and produced
**byte-identical** tokenizers — which is the check that this was actually
deterministic, rather than merely intended to be.

## 5. Training, and surviving a 12-hour wall

Kaggle gives ~30 GPU-hours a week per account and kills any session at 12 hours.
Each run needs ~10.9. There is no room for a crash, and a crash at hour 10 that
loses everything ends the project rather than delaying it.

What makes an 11-hour run survivable:

- **Atomic checkpoints.** Write to a temp file, `os.replace` onto the target, keep
  the previous as a fallback. A crash mid-write otherwise truncates the only copy.
- **RNG state saved with the weights.** Without it a resumed run replays the same
  batches it already learned from, and quietly trains on a shorter corpus than the
  log claims.
- **A wall-clock guard** (`--max-hours 11`) that checkpoints and exits cleanly, so
  *I* choose when the session ends instead of the platform choosing mid-write.
- **A runner that assumes it will die.** Re-running the entrypoint skips finished
  work and resumes partial work. It also detects the case where a run completed on
  the server but its output was never downloaded — which is what happens every
  time the laptop sleeps.
- **A lockfile**, added after two runners raced and pushed the same job 18 seconds
  apart, burning double the GPU on identical work.

Every one of those fired during this project. The last run made the case for all
of it: it stopped at step 55,000 instead of 60,000, the runner noticed, pushed a
continuation from the last checkpoint, finished the remaining 5,000 steps, and
moved on to scoring — with nobody awake.

Both corpora were trimmed to **exactly 995,000,000 tokens** before any training
started. If the arms differ in size, the ablation measures data quantity and
proves nothing.

## 6. The ablation — the filter worked

Both models were given the same eight committed Taglish prompts, 32 generations
each, scored by the same function that built the corpora.

| arm | seed 1 | seed 2 | spread |
|---|---|---|---|
| **filtered** — English fraction | 0.206 | 0.164 | 0.042 |
| **unfiltered** — English fraction | 0.061 | 0.070 | 0.009 |

The arms do not overlap. The gap is **0.120** — roughly three times as much
English — and **2.8× the widest within-arm spread**. Two seeds per arm is what
makes that sayable; with one run each I could not have separated it from noise.

What it sounds like. Filtered:

> Kahapon nag-iinom ako ng buko juice. Alam mo naman kasi yung feeling na gusto mo
> ng pakulo before ka mag-hard reservation. […] Maybe this time I'm going to be
> better off. Gets ko na lang, and I know I'm just hoping for that.

Unfiltered, same prompt:

> Kahapon nag-usap sila ni Jimmy sa telepono. Nang tanungin kung anong oras na ang
> kanilang komunikasyon para sa Skype, tinanong sila ni Jimmy kung ano ang sagot.

The difference is register, not vocabulary. The filtered model writes the way
someone texts — *"Gets ko na lang"*, *"Alam mo naman kasi yung feeling"* — flipping
into English for a whole clause and back. The control writes clean reported speech
in a news register. Both are fluent Filipino. Only one is Taglish.

### The weaker metric, reported anyway

My pre-registered headline was *Taglish share*, the fraction of generations
passing the full `is_taglish` test: **0.328** gap against a **0.219** within-arm
spread. Same direction, same conclusion, much thinner margin — because it is a
threshold count over 32 samples, so it throws away magnitude and inherits the
noise of a small binomial. The continuous measure separates cleanly; the
thresholded one barely clears. Reporting only the flattering one is how you fool
yourself.

### The number that looks like an answer and isn't

The unfiltered models reach val loss **2.635**; the filtered ones **2.745**. Lower
is better, so the control looks like the better model.

That reading is meaningless. **Each model is evaluated on a held-out slice of its
own corpus** — they sit different exams. Filtered Taglish is intrinsically harder
to predict: two languages of vocabulary, and a switch point that can fall
anywhere. It would have been very easy to put "2.635 vs 2.745" in a results table
and let a reader draw the wrong conclusion.

## 7. Can do / can't do

**Can:** fluent, register-correct conversational Taglish, switching mid-sentence.

**Can't:** anything requiring knowledge or symbol manipulation — and it will not
warn you.

> `Ang kabuuang populasyon ng Pilipinas noong 2020 ay` → **umabot sa 3.5 milyon.**

The real figure is about 109 million. Grammatical, plausibly shaped, off by a
factor of thirty.

> `1 + 1 =` → **0.04 – 5.05 = 0.48 – 0.8 – 0.8 – 0.2 – 0.8 …**

> `def fibonacci(n):` → **Southern California, Mexico, CA 9003, Mexico, Mexico …**

It degenerates into repetition at long horizons, invents names and dates fluently,
and has no instruction-following stage — it continues text, it does not answer
questions. Its idea of Taglish is the internet's: blogs, news comments,
entertainment copy.

**And it is under-trained.** Validation loss was still descending at step 60,000 in
all four runs — nothing flattened or turned up. The runs stopped because the weekly
GPU quota did, not because training converged. Both arms stopped at the same step,
so the comparison holds; the absolute quality does not represent a finished model.

## 8. What I'd do differently

**Measure the premise before designing around it.** The ~40% native code-switching
rate should have been the first thing I ran, not something I found after the
corpus design was locked. It would have changed the control arm — a genuinely
monolingual control, built by filtering *against* code-switching, would have given
a much sharper contrast than "unfiltered."

**Treat sample size as part of the number.** The 2,000-document bake-off cost me a
wrong claim in a draft. Any measurement small enough to run in a minute is small
enough to be backwards.

**Grep every reader before changing a config value.** I edited `max_steps` to pin
the learning-rate schedule and did not notice the pipeline used the same field to
decide whether a run had finished. A completed 60,000-step run looked 12,000 steps
short and got a pointless continuation. An assert at the top of the training
notebook caught it in **55 seconds** instead of hours of compute — cheap checks in
front of expensive work always pay.

**Deploy, then verify the remote copy.** I edited the configs and launched two
11-hour runs from the stale copy still sitting on the server. Both arms were
equally affected so the comparison survived, but the committed config no longer
described what ran. Editing a file is not shipping it.

**Scope every file glob.** Each saved run contains a full copy of the repo, so a
search for `pyproject.toml` matched both the repo and the previous run's embedded
copy, and broke a chain mid-flight. Scope the pattern so ambiguity is impossible
rather than merely detected.

**Say "checking" before naming a cause.** When mounted files first went missing I
blamed a race condition and said so with confidence. It was a wrong path — one
`ls` would have shown it. A confident wrong diagnosis costs more than an admitted
unknown.

## 9. Reproduce it

```bash
pip install -e .

python test_liitllm.py                  # transformer + training loop correctness
python test_taglish.py                  # the code-switch scorer
python test_data.py                     # corpus parity and dedup
python test_resume.py                   # a killed session can resume

python -m liitllm.data prepare          # stream → filter → tokenize (CPU, hours)
python -m liitllm.train --config configs/liit-33m.yaml
python -m liitllm.evaluate --baseline out/ckpt.pt --ablation out-unfiltered/ckpt.pt
```

The four gates are CPU-only and run in about a minute. They exist so that no GPU
hour is ever spent on a model that was broken before it started.

Free-GPU run: [notebooks/](notebooks/), generated by `build_notebooks.py`. The
full operational procedure — accounts, quotas, resuming a dead chain — is in
[RUNBOOK.md](RUNBOOK.md). The case study site in [docs/](docs/) is generated from
`results/` by `build_site.py`, which re-derives every quoted figure and fails the
build if the page and the artifacts disagree.

**Stack:** Python · PyTorch · HuggingFace `tokenizers` + `datasets` (streaming) ·
Kaggle free T4 · no pretrained weights, no paid services.
