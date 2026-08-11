"""Generate docs/index.html from the real artifacts.

    python docs/build_site.py

Every number, curve and lexicon on the page is read from results/ and
data/lexicons/ at build time. Nothing is transcribed by hand, so the page cannot
drift from what actually ran — re-run this after any new result and the site
updates itself.

Editorial choices (which generations to quote, what the prose says) live in this
file. Measurements do not.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "index.html"

RUNS = {
    "b1": "baseline-part1", "b2": "baseline2-part1",
    "a1": "ablation-part2", "a2": "ablation2-part2",
}


def lexicon(name: str) -> list[str]:
    words = []
    for line in (ROOT / "data" / "lexicons" / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip().lower()
        if line:
            words.append(line)
    return sorted(set(words))


def curves() -> dict:
    out = {}
    for key, slug in RUNS.items():
        rows = list(csv.DictReader((ROOT / "results" / slug / "out" / "loss.csv").open()))
        out[key] = [[int(r["step"]), round(float(r["val_loss"]), 4)]
                    for r in rows if int(r["step"]) % 2000 == 0]
    return out


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from liitllm import taglish as tg

    verdict = json.loads((ROOT / "results" / "verdict" / "codeswitch.json").read_text())
    data = {
        "tl": lexicon("tagalog"),
        "en": lexicon("english"),
        "pre": list(tg.PREFIXES),
        "suf": list(tg.SUFFIXES),
        "minTl": tg.MIN_TL, "minEn": tg.MIN_EN, "minWords": tg.MIN_WORDS,
        "curves": curves(),
        "verdict": verdict,
    }
    html = (ROOT / "docs" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*%%DATA%%*/", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")

    check(html, verdict)


def check(html: str, verdict: dict):
    """Fail loudly rather than publish a number the results do not support.

    The prose quotes figures the reader cannot verify. Each claim below is
    recomputed from the artifacts and matched against the string actually on the
    page, so a re-run with new results breaks the build instead of silently
    leaving stale numbers in the copy. This caught a real error: the page claimed
    the gap was 4x the within-arm spread when it is 2.8x.
    """
    arms = verdict["arms"]
    assert all(s["step"] == 60000 for a in arms.values() for s in a), "unequal training"
    assert verdict["taglish_share_gap"] > verdict["within_arm_spread"], "verdict flipped"

    def mean(k):
        v = [s["en_mean"] for s in arms[k]]
        return sum(v) / len(v)

    def spread(k):
        v = [s["en_mean"] for s in arms[k]]
        return max(v) - min(v)

    gap = mean("baseline_filtered") - mean("ablation_unfiltered")
    widest = max(spread("baseline_filtered"), spread("ablation_unfiltered"))
    ratio = mean("baseline_filtered") / mean("ablation_unfiltered")

    claims = [
        (f"{gap:.3f}", "en_mean gap"),
        (f"{ratio:.1f}", "arm ratio") if abs(ratio - 3) > 0.35 else ("three times", "arm ratio"),
        (f"{gap / widest:.1f}&times;", "gap / widest spread"),
        (f"{verdict['taglish_share_gap']:.3f}", "taglish share gap"),
        (f"{verdict['within_arm_spread']:.3f}", "within-arm spread"),
    ]
    for text, label in claims:
        assert text in html, f"page does not state {label} = {text}"
    for s in [x for a in arms.values() for x in a]:
        assert f"{s['en_mean']:.3f}" in html, f"missing en_mean {s['en_mean']:.3f}"
    print(f"  checked: gap {gap:.3f}, ratio {ratio:.2f}x, gap/spread {gap / widest:.2f}x")


if __name__ == "__main__":
    main()
