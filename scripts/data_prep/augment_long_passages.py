"""Amplify new_data.txt into training-grade rows and merge into final splits.

Strategy — turn 166 long passages into thousands of useful rows by emitting the
same content at multiple granularities so SGD actually sees it:

  1. Fix character-level typos (Hebrew/Persian lookalikes -> Arabic)
  2. Parse: passages separated by `## title`; paragraphs are non-empty,
     non-separator lines; drop `---` and title lines.
  3. Emit training rows at four granularities:
       a. FULL passage    = all paragraphs of a passage joined
       b. PARAGRAPH       = one paragraph
       c. MULTI-SENTENCE  = random 2-4 consecutive sentences
       d. SENTENCE        = one sentence
     Sentence split: only on . and ؟ (not commas/semicolons — those are
     intra-sentence in formal Arabic prose).
  4. Dedup everything against itself AND against the existing 184k rows in
     final_tashkeel_{train,eval}.jsonl.
  5. Stratify 90/10 into train/eval with fixed seed, write final_tashkeel_*_v2.jsonl.
  6. Print a stats report.

Source label on the new rows: `long_passages_v1`.
"""

import json
import os
import random
import re
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw")
SPLITS = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits")
SRC = os.path.join(RAW_DIR, "long_passages_raw.txt")
OLD_TRAIN = os.path.join(SPLITS, "final_tashkeel_train.jsonl")
OLD_EVAL = os.path.join(SPLITS, "final_tashkeel_eval.jsonl")
NEW_TRAIN = os.path.join(SPLITS, "final_tashkeel_train_v2.jsonl")
NEW_EVAL = os.path.join(SPLITS, "final_tashkeel_eval_v2.jsonl")
REPORT = os.path.join(ROOT, "docs", "long_passages_report.txt")

SEED = 42
EVAL_FRACTION = 0.10
USER_PREFIX = "أضف التشكيل:\n\n"
SOURCE = "long_passages_v1"

TASHKEEL_RE = re.compile(r"[ً-ْٰ]")
ARABIC_LETTER_RE = re.compile(r"[ء-ي]")

# Character-level fixes: lookalike letters from other scripts -> Arabic
CHAR_FIXES = {
    "ה": "ه",  # Hebrew he    -> Arabic heh
    "ک": "ك",  # Persian keheh -> Arabic kaf
    "ی": "ي",  # Persian yeh   -> Arabic yeh
    "ہ": "ه",  # Urdu heh      -> Arabic heh
    "ـ": "",        # tatweel: drop (CATT strips it anyway)
}


def apply_char_fixes(s):
    for bad, good in CHAR_FIXES.items():
        s = s.replace(bad, good)
    return s


def strip_tashkeel(s):
    return TASHKEEL_RE.sub("", s)


def dedup_key(s):
    return strip_tashkeel(s).replace(" ", "").replace("\n", "").strip()


def word_count(s):
    return len(strip_tashkeel(s).split())


def tashkeel_density(s):
    letters = len(ARABIC_LETTER_RE.findall(s))
    diacs = len(TASHKEEL_RE.findall(s))
    return (diacs / letters) if letters else 0.0


def parse_passages(text):
    """Yield (title, [paragraphs]) tuples."""
    # Normalize \r\n -> \n first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    current_title = None
    current_paragraphs = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            continue
        if line.startswith("## "):
            # flush previous passage
            if current_title is not None and current_paragraphs:
                yield current_title, current_paragraphs
            current_title = line[3:].strip()
            current_paragraphs = []
            continue
        current_paragraphs.append(line)
    if current_title is not None and current_paragraphs:
        yield current_title, current_paragraphs


def split_sentences(paragraph):
    """Split on sentence-terminal punctuation (. and ؟), keep the punctuation."""
    # Use a lookbehind-ish approach: split then re-attach terminators.
    parts = re.split(r"([.؟])", paragraph)
    sentences = []
    buf = ""
    for p in parts:
        if p in (".", "؟"):
            buf += p
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
        else:
            buf += p
    tail = buf.strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def make_row(text):
    plain = strip_tashkeel(text).strip()
    return {
        "source": SOURCE,
        "user": USER_PREFIX + plain,
        "assistant": text.strip(),
    }


def valid_row(row, min_words=4):
    if not row["assistant"].strip() or not row["user"].strip():
        return False
    if not TASHKEEL_RE.search(row["assistant"]):
        return False
    if word_count(row["assistant"]) < min_words:
        return False
    return True


def main():
    random.seed(SEED)

    with open(SRC, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = apply_char_fixes(raw)

    # Sanity: confirm fixes applied
    remaining_bad = {k: raw.count(k) for k in CHAR_FIXES if k != "ـ" and raw.count(k) > 0}

    passages = list(parse_passages(raw))
    print(f"Parsed {len(passages)} passages")

    rows_by_granularity = Counter()
    candidate_rows = []
    seen_keys = set()

    for title, paragraphs in passages:
        paragraphs = [apply_char_fixes(p).strip() for p in paragraphs]

        # A. FULL passage
        full = "\n\n".join(paragraphs)
        row = make_row(full)
        k = dedup_key(row["assistant"])
        if valid_row(row, min_words=50) and k not in seen_keys:
            seen_keys.add(k)
            candidate_rows.append(row)
            rows_by_granularity["full_passage"] += 1

        # B. PARAGRAPH
        for p in paragraphs:
            row = make_row(p)
            k = dedup_key(row["assistant"])
            if valid_row(row, min_words=10) and k not in seen_keys:
                seen_keys.add(k)
                candidate_rows.append(row)
                rows_by_granularity["paragraph"] += 1

        # C + D. sentences + multi-sentence chunks
        for p in paragraphs:
            sents = split_sentences(p)
            # D. single sentences
            for s in sents:
                row = make_row(s)
                k = dedup_key(row["assistant"])
                if valid_row(row, min_words=4) and k not in seen_keys:
                    seen_keys.add(k)
                    candidate_rows.append(row)
                    rows_by_granularity["sentence"] += 1
            # C. multi-sentence chunks: every 2-sentence window, every 3-sentence window
            for window in (2, 3, 4):
                for i in range(len(sents) - window + 1):
                    chunk = " ".join(sents[i:i + window])
                    row = make_row(chunk)
                    k = dedup_key(row["assistant"])
                    if valid_row(row, min_words=15) and k not in seen_keys:
                        seen_keys.add(k)
                        candidate_rows.append(row)
                        rows_by_granularity[f"chunk_{window}sent"] += 1

    print(f"\nAugmented row counts (before existing-corpus dedup):")
    for k in ["full_passage", "paragraph", "chunk_2sent", "chunk_3sent", "chunk_4sent", "sentence"]:
        print(f"  {k:<16} {rows_by_granularity[k]:>6}")
    print(f"  {'TOTAL':<16} {len(candidate_rows):>6}")

    # Dedup against existing corpus
    print(f"\nLoading existing rows to dedup against...")
    existing_keys = set()
    for path in [OLD_TRAIN, OLD_EVAL]:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                existing_keys.add(dedup_key(r.get("assistant", "")))
    print(f"  existing unique keys: {len(existing_keys):,}")

    before = len(candidate_rows)
    candidate_rows = [r for r in candidate_rows if dedup_key(r["assistant"]) not in existing_keys]
    removed = before - len(candidate_rows)
    print(f"  dropped {removed} rows that already exist in final_tashkeel_train/eval")
    print(f"  final new rows: {len(candidate_rows)}")

    # Length distribution
    wcs = sorted(word_count(r["assistant"]) for r in candidate_rows)
    if wcs:
        def p(x):
            return wcs[max(0, min(len(wcs)-1, int(x*(len(wcs)-1))))]
        print(f"\nWord-count of final new rows: min={wcs[0]} p10={p(0.1)} p50={p(0.5)} p90={p(0.9)} max={wcs[-1]}")

    # Bucket distribution of final new rows
    buckets = Counter()
    for r in candidate_rows:
        w = word_count(r["assistant"])
        if   w < 15:  buckets["<15"] += 1
        elif w < 30:  buckets["15-29"] += 1
        elif w < 50:  buckets["30-49"] += 1
        elif w < 100: buckets["50-99"] += 1
        elif w < 200: buckets["100-199"] += 1
        else:         buckets["200+"] += 1
    print("\nWord bucket distribution of final new rows:")
    for k in ["<15", "15-29", "30-49", "50-99", "100-199", "200+"]:
        print(f"  {k:<8} {buckets[k]:>6}")

    # Stratified 90/10 split
    random.shuffle(candidate_rows)
    k_eval = max(1, int(round(len(candidate_rows) * EVAL_FRACTION)))
    new_eval = candidate_rows[:k_eval]
    new_train = candidate_rows[k_eval:]

    # Load original splits, append new rows, write v2
    print(f"\nWriting v2 splits...")
    def load_jsonl(path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    out.append(json.loads(ln))
        return out

    old_train = load_jsonl(OLD_TRAIN)
    old_eval = load_jsonl(OLD_EVAL)

    final_train = old_train + new_train
    final_eval = old_eval + new_eval
    random.shuffle(final_train)
    random.shuffle(final_eval)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(NEW_TRAIN, final_train)
    write_jsonl(NEW_EVAL, final_eval)

    # Report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("Long Passages Augmentation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source file: {SRC}\n")
        f.write(f"Char fixes applied: {CHAR_FIXES}\n")
        f.write(f"Remaining non-Arabic lookalikes after fix: {remaining_bad}\n\n")
        f.write(f"Passages parsed: {len(passages)}\n")
        f.write(f"Rows by granularity (pre-dedup vs existing):\n")
        for k, v in rows_by_granularity.items():
            f.write(f"  {k:<18} {v}\n")
        f.write(f"\nFinal new rows (post-dedup): {len(candidate_rows)}\n")
        f.write(f"  -> train: {len(new_train)}\n")
        f.write(f"  -> eval : {len(new_eval)}\n\n")
        f.write(f"v2 split totals:\n")
        f.write(f"  final_tashkeel_train_v2.jsonl: {len(final_train)}\n")
        f.write(f"  final_tashkeel_eval_v2.jsonl : {len(final_eval)}\n")

    print(f"\nWrote:")
    print(f"  {NEW_TRAIN}  ({len(final_train)} rows)")
    print(f"  {NEW_EVAL}   ({len(final_eval)} rows)")
    print(f"  {REPORT}")
    print(f"\nSummary: {len(candidate_rows)} new long-passage rows injected, "
          f"source={SOURCE}. Train delta: +{len(new_train)}, Eval delta: +{len(new_eval)}.")


if __name__ == "__main__":
    main()
