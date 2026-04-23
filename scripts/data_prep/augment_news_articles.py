"""Inject scraped easyarabicnews.com articles into the canonical JSONLs.

Reads:
  datasets/arabic_tashkeel/raw/easyarabicnews_all.txt   (from scrape_easyarabicnews.py)

Produces training rows at multiple granularities (same strategy as
augment_long_passages.py: full article, paragraph, 2-3-4-sentence chunks, sentence),
dedups against the *current* canonical JSONLs, stratifies 90/10 train/eval, and
writes new canonical files with `_v3.jsonl` suffix (leaving current canonical
untouched for easy rollback).

Source label on the new rows: `easyarabicnews_v1`.

After this runs successfully, promote v3 -> canonical by renaming just like we
did for v2:
    final_tashkeel_train.jsonl -> final_tashkeel_train_pre_news.jsonl
    final_tashkeel_train_v3.jsonl -> final_tashkeel_train.jsonl
    (same for eval)
"""

import json
import os
import random
import re
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw")
SPLITS = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits")
SRC = os.path.join(RAW_DIR, "easyarabicnews_all.txt")
OLD_TRAIN = os.path.join(SPLITS, "final_tashkeel_train.jsonl")
OLD_EVAL = os.path.join(SPLITS, "final_tashkeel_eval.jsonl")
NEW_TRAIN = os.path.join(SPLITS, "final_tashkeel_train_v3.jsonl")
NEW_EVAL = os.path.join(SPLITS, "final_tashkeel_eval_v3.jsonl")
REPORT = os.path.join(ROOT, "docs", "news_articles_report.txt")

SEED = 42
EVAL_FRACTION = 0.10
USER_PREFIX = "أضف التشكيل:\n\n"
SOURCE = "easyarabicnews_v1"

TASHKEEL_RE = re.compile(r"[ً-ْٰ]")
ARABIC_LETTER_RE = re.compile(r"[ء-ي]")

# Character-level typo fixes for lookalike letters
CHAR_FIXES = {
    "ה": "ه",
    "ک": "ك",
    "ی": "ي",
    "ہ": "ه",
    "ـ": "",
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


def parse_passages(text):
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
            if current_title is not None and current_paragraphs:
                yield current_title, current_paragraphs
            current_title = line[3:].strip()
            current_paragraphs = []
            continue
        current_paragraphs.append(line)
    if current_title is not None and current_paragraphs:
        yield current_title, current_paragraphs


def split_sentences(paragraph):
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

    if not os.path.exists(SRC):
        print(f"SOURCE NOT FOUND: {SRC}")
        print("Run scripts/data_prep/scrape_easyarabicnews.py first.")
        return

    with open(SRC, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = apply_char_fixes(raw)

    passages = list(parse_passages(raw))
    print(f"Parsed {len(passages)} news articles from {os.path.basename(SRC)}")

    rows_by_granularity = Counter()
    candidate_rows = []
    seen_keys = set()

    for title, paragraphs in passages:
        paragraphs = [apply_char_fixes(p).strip() for p in paragraphs]

        # A. FULL article
        full = "\n\n".join(paragraphs)
        row = make_row(full)
        k = dedup_key(row["assistant"])
        if valid_row(row, min_words=30) and k not in seen_keys:
            seen_keys.add(k)
            candidate_rows.append(row)
            rows_by_granularity["full_article"] += 1

        # B. PARAGRAPH
        for p in paragraphs:
            row = make_row(p)
            k = dedup_key(row["assistant"])
            if valid_row(row, min_words=8) and k not in seen_keys:
                seen_keys.add(k)
                candidate_rows.append(row)
                rows_by_granularity["paragraph"] += 1

        # C + D. sentences + multi-sentence chunks
        for p in paragraphs:
            sents = split_sentences(p)
            for s in sents:
                row = make_row(s)
                k = dedup_key(row["assistant"])
                if valid_row(row, min_words=4) and k not in seen_keys:
                    seen_keys.add(k)
                    candidate_rows.append(row)
                    rows_by_granularity["sentence"] += 1
            for window in (2, 3, 4):
                for i in range(len(sents) - window + 1):
                    chunk = " ".join(sents[i:i + window])
                    row = make_row(chunk)
                    k = dedup_key(row["assistant"])
                    if valid_row(row, min_words=12) and k not in seen_keys:
                        seen_keys.add(k)
                        candidate_rows.append(row)
                        rows_by_granularity[f"chunk_{window}sent"] += 1

    print(f"\nAugmented row counts (before existing-corpus dedup):")
    for k in ["full_article", "paragraph", "chunk_2sent", "chunk_3sent", "chunk_4sent", "sentence"]:
        print(f"  {k:<16} {rows_by_granularity[k]:>6}")
    print(f"  {'TOTAL':<16} {len(candidate_rows):>6}")

    # Dedup against existing corpus
    print(f"\nLoading existing canonical to dedup against...")
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
    print(f"  dropped {removed} rows already in canonical")
    print(f"  final new news rows: {len(candidate_rows)}")

    # Length distribution
    wcs = sorted(word_count(r["assistant"]) for r in candidate_rows)
    if wcs:
        def p(x):
            return wcs[max(0, min(len(wcs)-1, int(x*(len(wcs)-1))))]
        print(f"\nWord-count of final new rows: min={wcs[0]} p10={p(0.1)} p50={p(0.5)} p90={p(0.9)} max={wcs[-1]}")

    # Bucket distribution
    buckets = Counter()
    for r in candidate_rows:
        w = word_count(r["assistant"])
        if   w < 15:  buckets["<15"] += 1
        elif w < 30:  buckets["15-29"] += 1
        elif w < 50:  buckets["30-49"] += 1
        elif w < 100: buckets["50-99"] += 1
        elif w < 200: buckets["100-199"] += 1
        else:         buckets["200+"] += 1
    print("\nNew news rows by word bucket:")
    for k in ["<15", "15-29", "30-49", "50-99", "100-199", "200+"]:
        print(f"  {k:<8} {buckets[k]:>6}")

    # Stratified 90/10 split
    random.shuffle(candidate_rows)
    k_eval = max(1, int(round(len(candidate_rows) * EVAL_FRACTION)))
    new_eval = candidate_rows[:k_eval]
    new_train = candidate_rows[k_eval:]

    # Load existing splits, append new rows, write v3
    print(f"\nWriting v3 splits...")

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

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("News Articles Augmentation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source file: {SRC}\n")
        f.write(f"Articles parsed: {len(passages)}\n")
        f.write(f"Augmented rows by granularity (pre-dedup):\n")
        for k, v in rows_by_granularity.items():
            f.write(f"  {k:<18} {v}\n")
        f.write(f"\nFinal new rows (post-dedup): {len(candidate_rows)}\n")
        f.write(f"  -> train: {len(new_train)}\n")
        f.write(f"  -> eval : {len(new_eval)}\n\n")
        f.write(f"v3 split totals:\n")
        f.write(f"  final_tashkeel_train_v3.jsonl: {len(final_train)}\n")
        f.write(f"  final_tashkeel_eval_v3.jsonl : {len(final_eval)}\n")

    print(f"\nWrote:")
    print(f"  {NEW_TRAIN}  ({len(final_train)} rows)")
    print(f"  {NEW_EVAL}   ({len(final_eval)} rows)")
    print(f"  {REPORT}")
    print(f"\nSummary: {len(candidate_rows)} new news rows injected, "
          f"source={SOURCE}. Train delta: +{len(new_train)}, Eval delta: +{len(new_eval)}.")


if __name__ == "__main__":
    main()
