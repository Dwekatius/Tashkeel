"""Prepare the local CSV dataset for CATT and punctuation training.

Input:
  datasets/arabic_tashkeel/raw/my_own_dataset_original.csv

Outputs:
  datasets/arabic_tashkeel/raw/my_own_dataset_clean.csv
  datasets/arabic_tashkeel/splits/own_dataset_v1_train.jsonl
  datasets/arabic_tashkeel/splits/own_dataset_v1_eval.jsonl
  datasets/arabic_tashkeel/splits/final_tashkeel_train_own_v1.jsonl
  datasets/arabic_tashkeel/splits/final_tashkeel_eval_own_v1.jsonl
  datasets/arabic_tashkeel/splits/final_tashkeel_own_v1_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import unicodedata
from collections import Counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw")
SPLITS_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits")

DEFAULT_INPUT = os.path.join(RAW_DIR, "my_own_dataset_original.csv")
DEFAULT_CLEAN_CSV = os.path.join(RAW_DIR, "my_own_dataset_clean.csv")
DEFAULT_BASE_TRAIN = os.path.join(SPLITS_DIR, "final_tashkeel_train.jsonl")
DEFAULT_BASE_EVAL = os.path.join(SPLITS_DIR, "final_tashkeel_eval.jsonl")
OWN_TRAIN = os.path.join(SPLITS_DIR, "own_dataset_v1_train.jsonl")
OWN_EVAL = os.path.join(SPLITS_DIR, "own_dataset_v1_eval.jsonl")
OUT_TRAIN = os.path.join(SPLITS_DIR, "final_tashkeel_train_own_v1.jsonl")
OUT_EVAL = os.path.join(SPLITS_DIR, "final_tashkeel_eval_own_v1.jsonl")
OUT_REPORT = os.path.join(SPLITS_DIR, "final_tashkeel_own_v1_report.txt")

USER_PREFIX = "أضف التشكيل:\n\n"
SOURCE_LABEL = "own_dataset_v1"
SEED = 20260424
EVAL_FRACTION = 0.10

TASHKEEL_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_RE = re.compile(r"[\u0621-\u064a\u0671]+")
CONTROL_RE = re.compile(r"[\ufeff\u200b-\u200f\u202a-\u202e]")
TRAILING_REF_RE = re.compile(r"(?:\s*\(\s*[\d٠-٩]+\s*/\s*[\d٠-٩]+\s*\)\s*)+$")
LEADING_REF_RE = re.compile(r"^\s*[\d٠-٩]+\s*[-–—]\s*")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛:؟!.])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([،؛:؟!.])(?=[\u0621-\u064a\u0671])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and merge the local CSV dataset.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--clean-csv", default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", default=DEFAULT_BASE_EVAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = CONTROL_RE.sub("", text)
    text = text.replace("ـ", "")
    text = text.replace(",", "،").replace(";", "؛").replace("?", "؟")
    text = LEADING_REF_RE.sub("", text)
    text = TRAILING_REF_RE.sub("", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text).replace("ـ", "")


def dedup_key(text: str) -> str:
    text = strip_tashkeel(text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_is_usable(cleaned_sentence: str, unvowel_sentence: str) -> tuple[bool, str]:
    if not cleaned_sentence or not unvowel_sentence:
        return False, "empty"
    if not ARABIC_RE.search(cleaned_sentence):
        return False, "no_arabic"
    if not TASHKEEL_RE.search(cleaned_sentence):
        return False, "no_tashkeel"
    if not ARABIC_RE.search(unvowel_sentence):
        return False, "no_unvowel_arabic"
    return True, "ok"


def load_existing_keys(*paths: str) -> set[str]:
    keys = set()
    for path in paths:
        for row in iter_jsonl(path):
            assistant = row.get("assistant", "")
            key = dedup_key(assistant)
            if key:
                keys.add(key)
    return keys


def make_jsonl_row(cleaned: dict) -> dict:
    return {
        "source": SOURCE_LABEL,
        "user": USER_PREFIX + cleaned["unvowel_sentence"],
        "assistant": cleaned["cleaned_sentence"],
        "metadata": {
            "filename": cleaned["filename"],
            "uuid": cleaned["uuid"],
        },
    }


def read_and_clean_csv(input_csv: str, existing_keys: set[str]) -> tuple[list[dict], dict]:
    stats = Counter()
    seen_own = set()
    rows: list[dict] = []

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "cleaned_sentence", "unvowel_sentence", "uuid"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        for row in reader:
            stats["input_rows"] += 1
            cleaned_sentence = normalize_text(row.get("cleaned_sentence", ""))
            unvowel_sentence = normalize_text(row.get("unvowel_sentence", ""))
            usable, reason = row_is_usable(cleaned_sentence, unvowel_sentence)
            if not usable:
                stats[f"skipped_{reason}"] += 1
                continue

            key = dedup_key(cleaned_sentence)
            if not key:
                stats["skipped_empty_key"] += 1
                continue
            if key in seen_own:
                stats["skipped_duplicate_within_csv"] += 1
                continue
            if key in existing_keys:
                stats["skipped_duplicate_existing_splits"] += 1
                continue

            seen_own.add(key)
            rows.append(
                {
                    "filename": normalize_text(row.get("filename", "")),
                    "uuid": normalize_text(row.get("uuid", "")),
                    "cleaned_sentence": cleaned_sentence,
                    "unvowel_sentence": unvowel_sentence,
                }
            )
            stats["usable_rows"] += 1

    stats["unique_keys"] = len(seen_own)
    return rows, dict(stats)


def write_clean_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["filename", "uuid", "split", "cleaned_sentence", "unvowel_sentence"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def split_rows(rows: list[dict], seed: int, eval_fraction: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_fraction)) if shuffled else 0
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    for row in train_rows:
        row["split"] = "train"
    for row in eval_rows:
        row["split"] = "eval"
    return train_rows, eval_rows


def append_rows(base_path: str, extra_rows: list[dict], out_path: str) -> int:
    count = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for row in iter_jsonl(base_path):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        for row in extra_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def punctuation_counts(rows: list[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        for char in row["cleaned_sentence"]:
            if char in "،.؟؛:!":
                counts[char] += 1
    return counts


def write_report(
    path: str,
    stats: dict,
    train_rows: list[dict],
    eval_rows: list[dict],
    combined_counts: dict,
    seed: int,
    eval_fraction: float,
) -> None:
    punct = punctuation_counts(train_rows + eval_rows)
    word_counts = [len(row["cleaned_sentence"].split()) for row in train_rows + eval_rows]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Own dataset v1 import report\n")
        handle.write("============================\n\n")
        handle.write(f"Source label: {SOURCE_LABEL}\n")
        handle.write(f"Seed: {seed}\n")
        handle.write(f"Eval fraction: {eval_fraction}\n\n")
        handle.write("CSV cleanup stats\n")
        for key, value in sorted(stats.items()):
            handle.write(f"  {key:<36} {value}\n")
        handle.write("\n")
        handle.write(f"Own train rows: {len(train_rows)}\n")
        handle.write(f"Own eval rows : {len(eval_rows)}\n")
        if word_counts:
            handle.write(
                f"Word counts   : min={min(word_counts)} avg={sum(word_counts) / len(word_counts):.2f} max={max(word_counts)}\n"
            )
        handle.write("\nPunctuation in own usable rows\n")
        for char, count in punct.most_common():
            handle.write(f"  {char} {count}\n")
        handle.write("\nCombined outputs\n")
        for key, value in combined_counts.items():
            handle.write(f"  {key:<36} {value}\n")


def main() -> None:
    args = parse_args()
    existing_keys = load_existing_keys(args.base_train, args.base_eval)
    cleaned_rows, stats = read_and_clean_csv(args.input_csv, existing_keys)
    train_clean, eval_clean = split_rows(cleaned_rows, args.seed, args.eval_fraction)
    clean_with_split = train_clean + eval_clean

    train_jsonl = [make_jsonl_row(row) for row in train_clean]
    eval_jsonl = [make_jsonl_row(row) for row in eval_clean]

    write_clean_csv(args.clean_csv, clean_with_split)
    write_jsonl(OWN_TRAIN, train_jsonl)
    write_jsonl(OWN_EVAL, eval_jsonl)

    combined_counts = {
        "combined_train_rows": append_rows(args.base_train, train_jsonl, OUT_TRAIN),
        "combined_eval_rows": append_rows(args.base_eval, eval_jsonl, OUT_EVAL),
        "own_train_path": OWN_TRAIN,
        "own_eval_path": OWN_EVAL,
        "combined_train_path": OUT_TRAIN,
        "combined_eval_path": OUT_EVAL,
        "clean_csv_path": args.clean_csv,
    }
    write_report(OUT_REPORT, stats, train_clean, eval_clean, combined_counts, args.seed, args.eval_fraction)

    print("Own dataset v1 prepared.")
    print(f"Clean CSV:       {args.clean_csv}")
    print(f"Own train JSONL: {OWN_TRAIN} ({len(train_jsonl)} rows)")
    print(f"Own eval JSONL:  {OWN_EVAL} ({len(eval_jsonl)} rows)")
    print(f"Combined train:  {OUT_TRAIN}")
    print(f"Combined eval:   {OUT_EVAL}")
    print(f"Report:          {OUT_REPORT}")


if __name__ == "__main__":
    main()
