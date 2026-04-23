"""Quick check + clean of 'new data.txt'.

Goal: long-passage Arabic tashkeel data, targeted at 50-500 word paragraphs.
Format observed: passages separated by `## <title>`, each followed by blank
lines and multiple paragraphs. One paragraph per non-empty line.

Reports:
  - passage count (## headers)
  - paragraph count and word-count distribution
  - tashkeel density (fraction of Arabic letters that carry a diacritic)
  - non-Arabic garbage (latin letters, digits outside normal ranges, stray chars)
  - duplicates within this file
  - overlap vs our existing final_tashkeel_train/eval JSONLs
  - would-be row counts if added as 1-row-per-paragraph training samples
"""

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEW_PATH = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw", "long_passages_raw.txt")
TRAIN_JSONL = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_train.jsonl")
EVAL_JSONL = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_eval.jsonl")

TASHKEEL_RE = re.compile(r"[ً-ْٰ]")
ARABIC_LETTER_RE = re.compile(r"[ء-ي]")
# CATT's allowed char set (Arabic letters, diacritics, shaddah, ligatures, space)
CATT_KEEP_RE = re.compile(r"[^ء-غـ-ْٰٱﻻﻷﻵﻹ ]")


def strip_tashkeel(s):
    return TASHKEEL_RE.sub("", s)


def dedup_key(s):
    return strip_tashkeel(s).replace(" ", "").replace("\n", "").strip()


def tashkeel_density(s):
    letters = len(ARABIC_LETTER_RE.findall(s))
    diacs = len(TASHKEEL_RE.findall(s))
    return (diacs / letters) if letters else 0.0


def word_count(s):
    return len(strip_tashkeel(s).split())


def main():
    if not os.path.exists(NEW_PATH):
        print(f"NOT FOUND: {NEW_PATH}")
        sys.exit(1)

    with open(NEW_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    # Count passages by ## headers
    headers = [ln for ln in raw.splitlines() if ln.lstrip().startswith("## ")]
    print(f"File size: {len(raw):,} chars, {len(raw.encode('utf-8')):,} bytes")
    print(f"Passages (## headers): {len(headers)}")

    # Extract paragraphs: any non-empty, non-header line
    paragraphs = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("## "):
            continue
        paragraphs.append(s)
    print(f"Paragraphs (non-empty, non-header): {len(paragraphs)}")

    # Word count distribution
    wcs = [word_count(p) for p in paragraphs]
    wcs_sorted = sorted(wcs)
    def pct(p):
        if not wcs_sorted:
            return 0
        i = max(0, min(len(wcs_sorted) - 1, int(round(p * (len(wcs_sorted) - 1)))))
        return wcs_sorted[i]
    print(f"\nWord count per paragraph:")
    print(f"  min={min(wcs)}  p10={pct(0.10)}  p50={pct(0.50)}  p90={pct(0.90)}  max={max(wcs)}")
    print(f"  mean={sum(wcs)/len(wcs):.1f}")

    # Target band check
    in_band = sum(1 for w in wcs if 50 <= w <= 500)
    short = sum(1 for w in wcs if w < 50)
    long_ = sum(1 for w in wcs if w > 500)
    print(f"  in target band (50-500 words): {in_band}/{len(wcs)} ({in_band/len(wcs)*100:.1f}%)")
    print(f"  shorter than 50 words: {short}")
    print(f"  longer than 500 words: {long_}")

    # Tashkeel density
    densities = [tashkeel_density(p) for p in paragraphs]
    low_dens = [i for i, d in enumerate(densities) if d < 0.6]
    print(f"\nTashkeel density (diacs / Arabic letters):")
    print(f"  mean={sum(densities)/len(densities):.3f}  min={min(densities):.3f}  max={max(densities):.3f}")
    print(f"  paragraphs below 0.6 density: {len(low_dens)}")
    if low_dens[:3]:
        for i in low_dens[:3]:
            preview = paragraphs[i][:80]
            print(f"    idx {i} (dens {densities[i]:.2f}): {preview!r}")

    # Non-CATT chars
    bad_chars_counter = Counter()
    for p in paragraphs:
        for ch in p:
            if CATT_KEEP_RE.match(ch):
                bad_chars_counter[ch] += 1
    print(f"\nNon-CATT chars found (will be stripped on clean):")
    if bad_chars_counter:
        for ch, n in bad_chars_counter.most_common(20):
            print(f"  {ch!r}  U+{ord(ch):04X}  x{n}")
    else:
        print("  (none)")

    # Duplicates within this file
    keys = [dedup_key(p) for p in paragraphs]
    c = Counter(keys)
    dupes = [k for k, n in c.items() if n > 1]
    print(f"\nInternal duplicates: {len(dupes)} duplicate keys ({sum(c[k] for k in dupes)} total rows involved)")

    # Overlap with existing training data
    print(f"\nChecking overlap with existing train/eval JSONLs...")
    existing_keys = set()
    for path in [TRAIN_JSONL, EVAL_JSONL]:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                existing_keys.add(dedup_key(row.get("assistant", "")))
                n += 1
        print(f"  {os.path.basename(path)}: {n:,} rows loaded")
    new_keys = set(keys)
    overlap = new_keys & existing_keys
    print(f"  overlap (paragraphs already in train/eval): {len(overlap)}")

    # If we dropped dupes + overlap + out-of-band, what would we have?
    seen = set()
    final_rows = []
    for p, k, w in zip(paragraphs, keys, wcs):
        if k in existing_keys:
            continue
        if k in seen:
            continue
        seen.add(k)
        final_rows.append((p, w))
    print(f"\nPotential final rows after dedup + overlap removal:")
    print(f"  total: {len(final_rows)}")
    final_wcs = sorted([w for _, w in final_rows])
    if final_wcs:
        print(f"  word count: min={final_wcs[0]} p50={final_wcs[len(final_wcs)//2]} max={final_wcs[-1]}")

    # Compare to current long-row coverage in train
    print(f"\nCurrent long-row coverage in train set (for reference):")
    if os.path.exists(TRAIN_JSONL):
        long_by_bucket = Counter()
        total = 0
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for ln in f:
                row = json.loads(ln)
                w = word_count(row["assistant"])
                total += 1
                if w >= 50:
                    long_by_bucket["50-99"] += int(50 <= w < 100)
                    long_by_bucket["100-199"] += int(100 <= w < 200)
                    long_by_bucket["200-499"] += int(200 <= w < 500)
                    long_by_bucket["500+"] += int(w >= 500)
        print(f"  total rows: {total:,}")
        for k in ["50-99", "100-199", "200-499", "500+"]:
            v = long_by_bucket[k]
            print(f"  {k} words: {v:,} ({v/total*100:.2f}%)")

    # New data coverage by bucket
    print(f"\nNew data coverage by bucket:")
    buck = Counter()
    for _, w in final_rows:
        if w < 50: buck["<50"] += 1
        elif w < 100: buck["50-99"] += 1
        elif w < 200: buck["100-199"] += 1
        elif w < 500: buck["200-499"] += 1
        else: buck["500+"] += 1
    for k in ["<50", "50-99", "100-199", "200-499", "500+"]:
        print(f"  {k}: {buck[k]}")


if __name__ == "__main__":
    main()
