"""Score a document for Tagalog-English code-switching, and keep it or drop it.

This is the module the whole project turns on.

Every large web corpus — CC-100, OSCAR, FineWeb-2 — is assembled by a
document-level language-ID filter, and those filters treat code-switching as
low-confidence noise: a Filipino document with heavy English either falls below
the confidence threshold and is dropped, or gets labelled `en` and lands in the
English bucket. Standard pipelines filter code-switching *out*. This one filters
*for* it.

How the score works
-------------------
Two lexicons, two fractions. `tl_frac` is the share of word tokens found in a
Tagalog list weighted toward function words (markers, pronouns, particles);
`en_frac` is the share found in a high-frequency English list that deliberately
includes content words. A document is kept when **both** are meaningfully
non-zero — that is what "code-switched" means operationally.

Why function words carry the Tagalog side: they are a closed class, they are the
most frequent tokens in any real sentence, and a writer cannot produce Tagalog
syntax without them. Why the English side needs content words: the signature
form of Taglish borrows English *content* into Tagalog syntax — "nag-commute ako
sa office, na-late pa rin sa meeting" — where the English contribution is almost
entirely nouns and verbs. Scoring English on function words alone would rate that
sentence near 0% English and discard the exact documents we are looking for.

Two details that matter more than they look
-------------------------------------------
*Hyphens.* Tagalog affixes attach to English roots across a hyphen —
"nag-commute", "na-cancel", "mag-download". Splitting on the hyphen yields a
Tagalog affix and an English root, so the single most characteristic Taglish
construction scores on both sides instead of neither.

*Affixes.* Tagalog is agglutinative: "trabaho" appears as nagtrabaho,
magtrabaho, pinagtrabahuhan. Matching bare roots alone would undercount `tl_frac`
badly and reject good Taglish, so a miss retries after stripping common affixes.

Thresholds were set by reading sampled documents in each band, not derived. Rerun
`python -m liitllm.taglish sample <file>` after any lexicon change and re-read
them — this scorer decides the training corpus, the ablation, *and* the headline
metric, so a silent regression here is invisible in all three at once.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"

# Kept when tl_frac >= MIN_TL and en_frac >= MIN_EN. Both fractions are shares of
# all word tokens, so they do not sum to 1 — most tokens are in neither list.
MIN_TL = 0.10   # below this the document is not anchored in Tagalog grammar
MIN_EN = 0.06   # below this it is monolingual Tagalog, not code-switched
MIN_WORDS = 20  # shorter than this the fractions are noise, not measurement

# Common Tagalog affixes, longest first so "nakaka" is tried before "na".
PREFIXES = (
    "pinakama", "pinagpa", "nakakapag", "makakapag", "nagpapa", "magpapa",
    "nakaka", "makaka", "ipinag", "napaka", "pinaka", "nagpa", "magpa",
    "pinag", "nakag", "makag", "ipina", "naka", "maka", "ipag", "pina",
    "nagka", "magka", "nang", "pag", "nag", "mag", "pan", "pam", "ma",
    "na", "ka", "pa", "i",
)
SUFFIXES = ("ohan", "uhan", "han", "hin", "ang", "ing", "an", "in", "ng", "g")

_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


@lru_cache(maxsize=None)
def _lexicon(name: str) -> frozenset[str]:
    path = LEXICON_DIR / f"{name}.txt"
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return frozenset(words)


def words(text: str) -> list[str]:
    """Word tokens, lowercased, with hyphenated compounds split into parts.

    The split is the point: "nag-commute" becomes ["nag", "commute"], which
    scores on the Tagalog side *and* the English side. Kept as one token it would
    match neither list and the most distinctively Taglish word in the sentence
    would contribute nothing.
    """
    out = []
    for match in _WORD_RE.findall(text.lower()):
        out.extend(p for p in match.split("-") if p)
    return out


def _is_tagalog(word: str, lex: frozenset[str]) -> bool:
    """Direct hit, or a hit after peeling one affix off each end.

    One prefix and one suffix covers the overwhelming majority of real forms
    (nagtrabaho, trabahuhan, pinagtrabahuhan). Deeper stacking exists but each
    extra layer costs precision — "nasa" would reduce to "sa" — so this stops at
    one.
    ponytail: single-layer stripping; add a real morphological analyser only if
    the sampled output shows it changing decisions, not just scores.
    """
    if word in lex:
        return True
    if len(word) < 4:
        return False
    for p in PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3:
            stem = word[len(p):]
            if stem in lex:
                return True
            # Reduplication: nagcocommute -> stem "cocommute" -> "commute".
            if len(stem) > 2 and stem[:2] == stem[2:4] and stem[2:] in lex:
                return True
            for s in SUFFIXES:
                if stem.endswith(s) and stem[: -len(s)] in lex:
                    return True
    for s in SUFFIXES:
        if word.endswith(s) and len(word) - len(s) >= 3 and word[: -len(s)] in lex:
            return True
    return False


def score(text: str) -> tuple[float, float]:
    """Return (tl_frac, en_frac): the share of word tokens matching each lexicon."""
    toks = words(text)
    if not toks:
        return 0.0, 0.0
    tl_lex, en_lex = _lexicon("tagalog"), _lexicon("english")
    tl = en = 0
    for w in toks:
        # Tagalog first: the lists overlap on a few short strings, and Tagalog
        # membership is the more specific claim.
        if _is_tagalog(w, tl_lex):
            tl += 1
        elif w in en_lex:
            en += 1
    n = len(toks)
    return tl / n, en / n


def is_taglish(text: str, min_tl: float = MIN_TL, min_en: float = MIN_EN,
               min_words: int = MIN_WORDS) -> bool:
    """Keep documents where both languages are meaningfully present.

    Rejects, in the three ways a document can fail: pure English (no Tagalog
    anchor), monolingual Tagalog (no English), and anything too short for the
    fractions to mean anything — a 6-word snippet can hit any ratio by accident.
    """
    if len(words(text)) < min_words:
        return False
    tl, en = score(text)
    return tl >= min_tl and en >= min_en


def _demo():
    """Self-check: the three document types must land in the right bands."""
    english = (
        "The company announced today that it will open a new office in the city "
        "next year. The plan is expected to create more jobs for local people "
        "and the government said it would support the project with funding."
    )
    tagalog = (
        "Ang mga bata ay naglalaro sa labas ng bahay tuwing hapon. Masaya sila "
        "dahil maganda ang panahon at walang ulan. Sabi ng kanilang nanay, "
        "kailangan nilang umuwi bago dumilim ang langit sa gabi."
    )
    taglish = (
        "Grabe, nag-commute ako kanina from Makati papuntang office tapos na-late "
        "pa rin ako sa meeting. Ang traffic talaga sobrang hassle, kaya next time "
        "mag-book na lang siguro ako ng grab kesa mag-bus pa."
    )

    en_tl, en_en = score(english)
    tg_tl, tg_en = score(tagalog)
    tx_tl, tx_en = score(taglish)
    print(f"  english  tl={en_tl:.2f} en={en_en:.2f}  keep={is_taglish(english)}")
    print(f"  tagalog  tl={tg_tl:.2f} en={tg_en:.2f}  keep={is_taglish(tagalog)}")
    print(f"  taglish  tl={tx_tl:.2f} en={tx_en:.2f}  keep={is_taglish(taglish)}")

    assert not is_taglish(english), "pure English was kept"
    assert not is_taglish(tagalog), "monolingual Tagalog was kept"
    assert is_taglish(taglish), "real Taglish was rejected"
    assert not is_taglish("Grabe ang traffic sa EDSA today"), "short text was kept"
    # The signature construction must score on both sides, not neither.
    tl, en = score("nag-commute ako sa office")
    assert tl > 0 and en > 0, f"hyphenated code-switch scored tl={tl} en={en}"
    print("  ok  all bands correct")


def _sample(path: str, n: int = 50):
    """Print kept and rejected documents so the thresholds can be eyeballed.

    Reading the output is not optional after a lexicon change: a bad list shifts
    every score without ever raising an error.
    """
    kept, dropped = [], []
    for line in Path(path).read_text(encoding="utf-8").split("\n\n"):
        line = line.strip()
        if not line:
            continue
        (kept if is_taglish(line) else dropped).append(line)
    total = len(kept) + len(dropped)
    print(f"kept {len(kept)}/{total} ({100 * len(kept) / max(total, 1):.1f}%)\n")
    for label, docs in (("KEPT", kept), ("DROPPED", dropped)):
        print(f"===== {label} =====")
        for d in docs[:n]:
            tl, en = score(d)
            print(f"[tl={tl:.2f} en={en:.2f}] {d[:200]}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["demo", "sample"])
    p.add_argument("path", nargs="?")
    a = p.parse_args()
    if a.cmd == "demo":
        _demo()
    else:
        _sample(a.path)
