# LiitLLM — Handoff / Kickoff Context

> A tiny **Tagalog** language model built from scratch, reusing the Munti
> codebase. This is a handoff only — the detailed plan is intentionally left for
> the owner to write. This file gives a fresh session enough context to help
> plan and build from zero.

## What this is

**LiitLLM** (Filipino *liit*: small) — a small, from-scratch language model
trained on **Tagalog** text on free hardware, until it produces coherent
Filipino text. It is the language sibling of **Munti** (the TinyStories
from-scratch model) and is meant to **reuse Munti's language-agnostic codebase**
almost unchanged — the new work is data, not architecture.

One-sentence identity: **"I built a language model for Filipino, from scratch,
on free hardware"** — the most distinctive item on the portfolio.

## Who it's for / who's building it

- Owner: John Andrei Martinez (GitHub `Zeref538`), AI/ML student, portfolio at
  johnandrei.vercel.app. Becomes a portfolio card + case study.
- Learning artifact, not a product — it will not be a useful chatbot, and that's
  stated openly. Value = demonstrated understanding + a culturally distinct
  result + honest evaluation.

## Depends on Munti

Build **Munti first** (`Portfolio/Munti/`). LiitLLM's whole premise is that the
transformer, training loop, tokenizer, and sampling already exist and are proven
correct on TinyStories — so any rough Tagalog output is about the *data/language*,
not a code bug. If Munti isn't done, do it before starting here.

## Non-negotiable constraints (hard requirements)

1. **From scratch.** Same as Munti — write the components yourself; no
   pretrained models, no prebuilt GPT class.
2. **₱0 cost.** Free Kaggle GPU, free/open Tagalog corpora, free static host.
3. **Tiny + trainable in hours** (~10–35M params, one free Kaggle session).
4. **Reuse Munti's codebase** — architecture and training loop unchanged; the
   real work is the tokenizer (may need to be retrained for Tagalog) and the
   data pipeline.
5. **Honest framing** — explicit can-do / can't-do section grounded in real
   samples.

## The hard part is DATA (this is what to plan carefully)

Unlike Munti (TinyStories is clean and ready), Tagalog data must be gathered and
**cleaned** — that's the bulk of the effort. Candidate free/open sources to
evaluate when planning:

- Tagalog **Wikipedia** dump (free, sizable, needs markup cleaning)
- **OSCAR** / **CC-100** Tagalog subsets (web-crawled — noisier, needs filtering)
- **Leipzig Corpora Collection** — Tagalog news/web sentences
- Any public-domain Filipino text (e.g. classic literature)

Known data challenges to plan around: Taglish code-switching, inconsistent
spelling/diacritics, web noise, and deduplication. The tokenizer choice
(char-level vs. a Tagalog-trained BPE) matters more here than in Munti.

## Kickoff prompt (paste into the fresh session, once YOU'VE planned it)

> I'm building LiitLLM, a tiny Tagalog language model from scratch, reusing my
> Munti codebase (a from-scratch PyTorch transformer already proven on
> TinyStories). Read HANDOFF.md in this folder for context. Hard constraints:
> from scratch (no pretrained models), free tier only (Kaggle GPU + open Tagalog
> corpora + free host), ~10–35M params, honest can-do/can't-do framing. Reuse
> Munti's model + training loop unchanged; the real work is Tagalog data
> sourcing/cleaning and the tokenizer. Here's my plan: [OWNER FILLS THIS IN].
> Help me execute it, starting with the data pipeline.

## First moves (when you're ready)

1. Finish Munti and confirm its code is reusable/language-agnostic.
2. Copy Munti's codebase in; retrain/replace the tokenizer for Tagalog.
3. Source + clean a Tagalog corpus (the main effort — plan this deliberately).
4. Train on the free GPU, capture loss curve + samples, write the honest
   case study, add the portfolio card.

## Definition of done (v1)

A from-scratch model that produces coherent Tagalog continuations from prompts;
committed loss curve + samples; an honest can-do/can't-do writeup that notes what
a tiny model captures in Filipino (grammar, common phrasing) and what it misses;
and a portfolio card in the house format.

## Note

Detailed PRD/PLAN intentionally **not written** — the owner is planning this one
themselves. This handoff is context only.
