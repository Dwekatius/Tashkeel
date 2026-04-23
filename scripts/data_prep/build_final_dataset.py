"""Build the final optimized training dataset.

Combines:
  - consolidated_dataset.txt (ambiguous-word pairs, 11 tiers + long paragraphs)
  - multiturn_tashkeel_train.jsonl (Quran + Tashkeela + parquet sources)
  - multiturn_tashkeel_eval.jsonl

Output: final_tashkeel_train.jsonl + final_tashkeel_eval.jsonl
Format: {"source": "<label>", "user": "أضف التشكيل:\n\n<plain>", "assistant": "<tashkeel>"}

Optimization:
  - Global dedup by tashkeel-stripped, space-stripped assistant text.
  - Existing multiturn train/eval split preserved (keeps eval stable for metrics).
  - New ambiguous-word data: stratified 90/10 per tier so eval covers every tier.
  - Quality filters: drop rows with empty/whitespace-only fields, drop rows where
    assistant contains zero tashkeel marks (useless for training).
  - Shuffle each split with fixed seed for reproducibility.
  - Per-source row counts reported.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw")
CONS_FILE = os.path.join(RAW_DIR, "consolidated_ambiguous_pairs.txt")
SPLITS_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits")
SRC_TRAIN = os.path.join(SPLITS_DIR, "multiturn_tashkeel_train.jsonl")
SRC_EVAL = os.path.join(SPLITS_DIR, "multiturn_tashkeel_eval.jsonl")
OUT_TRAIN = os.path.join(SPLITS_DIR, "final_tashkeel_train.jsonl")
OUT_EVAL = os.path.join(SPLITS_DIR, "final_tashkeel_eval.jsonl")
REPORT_FILE = os.path.join(ROOT, "docs", "final_dataset_report.txt")

USER_PREFIX = "أضف التشكيل:\n\n"
SEED = 42
EVAL_FRACTION = 0.10

TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def strip_tashkeel(s: str) -> str:
    return TASHKEEL_RE.sub("", s)


def dedup_key(s: str) -> str:
    return strip_tashkeel(s).replace(" ", "").replace("\n", "").strip()


def parse_consolidated():
    """Yield (tier_label, sentence_with_tashkeel) from consolidated_dataset.txt."""
    current_tier = "unknown"
    with open(CONS_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue
            if s.startswith("===="):
                continue
            if s.startswith("# Consolidated") or s.startswith("# Format") or s.startswith("# Grouped"):
                continue
            if s.startswith("# Tier") or s.startswith("# Bonus"):
                m = re.match(r"#\s*(Tier\s*\d+|Bonus)", s)
                if m:
                    current_tier = m.group(1).lower().replace(" ", "")
                continue
            if s.startswith("## "):
                continue
            sentence = s.split("|", 1)[0].strip() if "|" in s else s
            if not sentence:
                continue
            yield f"ambiguous_{current_tier}", sentence


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def row_is_valid(row) -> bool:
    if not all(k in row for k in ("source", "user", "assistant")):
        return False
    if not row["assistant"].strip() or not row["user"].strip():
        return False
    if not TASHKEEL_RE.search(row["assistant"]):
        return False
    return True


def make_row(source, sentence):
    plain = strip_tashkeel(sentence).strip()
    return {
        "source": source,
        "user": USER_PREFIX + plain,
        "assistant": sentence.strip(),
    }


def main():
    random.seed(SEED)
    seen_keys = set()
    train_rows = []
    eval_rows = []

    stats = {
        "existing_train_loaded": 0,
        "existing_eval_loaded": 0,
        "existing_dupes": 0,
        "existing_invalid": 0,
        "new_added": 0,
        "new_dupes_vs_existing": 0,
        "new_invalid": 0,
    }

    # 1. Existing multiturn — preserve split assignment
    for row in load_jsonl(SRC_TRAIN):
        if not row_is_valid(row):
            stats["existing_invalid"] += 1
            continue
        k = dedup_key(row["assistant"])
        if k in seen_keys:
            stats["existing_dupes"] += 1
            continue
        seen_keys.add(k)
        train_rows.append(row)
        stats["existing_train_loaded"] += 1

    for row in load_jsonl(SRC_EVAL):
        if not row_is_valid(row):
            stats["existing_invalid"] += 1
            continue
        k = dedup_key(row["assistant"])
        if k in seen_keys:
            stats["existing_dupes"] += 1
            continue
        seen_keys.add(k)
        eval_rows.append(row)
        stats["existing_eval_loaded"] += 1

    # 2. New ambiguous-word rows — bucket per tier for stratified split
    per_tier_new = defaultdict(list)
    for source, sentence in parse_consolidated():
        row = make_row(source, sentence)
        if not row_is_valid(row):
            stats["new_invalid"] += 1
            continue
        k = dedup_key(row["assistant"])
        if not k:
            stats["new_invalid"] += 1
            continue
        if k in seen_keys:
            stats["new_dupes_vs_existing"] += 1
            continue
        seen_keys.add(k)
        per_tier_new[source].append(row)
        stats["new_added"] += 1

    # 3. Stratified 90/10 split per tier
    new_train_total = 0
    new_eval_total = 0
    for tier, rows in per_tier_new.items():
        random.shuffle(rows)
        k = max(1, int(round(len(rows) * EVAL_FRACTION)))
        eval_slice = rows[:k]
        train_slice = rows[k:]
        eval_rows.extend(eval_slice)
        train_rows.extend(train_slice)
        new_eval_total += len(eval_slice)
        new_train_total += len(train_slice)

    # 4. Shuffle final splits
    random.shuffle(train_rows)
    random.shuffle(eval_rows)

    # 5. Write
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. Report
    train_sources = Counter(r["source"] for r in train_rows)
    eval_sources = Counter(r["source"] for r in eval_rows)
    train_chars = sum(len(r["assistant"]) for r in train_rows)
    eval_chars = sum(len(r["assistant"]) for r in eval_rows)

    with open(REPORT_FILE, "w", encoding="utf-8") as rep:
        rep.write("Final Optimized Training Dataset\n")
        rep.write("=" * 50 + "\n\n")
        rep.write(f"Seed                       : {SEED}\n")
        rep.write(f"Eval fraction (new data)   : {EVAL_FRACTION}\n")
        rep.write(f"Eval split strategy        : preserve existing + stratified per tier\n\n")
        rep.write("Load stats:\n")
        for k, v in stats.items():
            rep.write(f"  {k:<30} {v:>7}\n")
        rep.write(f"  new_to_train                 {new_train_total:>7}\n")
        rep.write(f"  new_to_eval                  {new_eval_total:>7}\n\n")
        rep.write("Final row counts:\n")
        rep.write(f"  TRAIN rows  : {len(train_rows)}\n")
        rep.write(f"  EVAL  rows  : {len(eval_rows)}\n")
        rep.write(f"  TOTAL rows  : {len(train_rows) + len(eval_rows)}\n\n")
        rep.write("Assistant char totals (proxy for training volume):\n")
        rep.write(f"  TRAIN chars : {train_chars:,}\n")
        rep.write(f"  EVAL  chars : {eval_chars:,}\n\n")
        rep.write("Train source breakdown:\n")
        for src, cnt in train_sources.most_common():
            rep.write(f"  {src:<30} {cnt:>7}\n")
        rep.write("\nEval source breakdown:\n")
        for src, cnt in eval_sources.most_common():
            rep.write(f"  {src:<30} {cnt:>7}\n")

    print(f"TRAIN: {len(train_rows)} rows -> {OUT_TRAIN}")
    print(f"EVAL : {len(eval_rows)} rows -> {OUT_EVAL}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
