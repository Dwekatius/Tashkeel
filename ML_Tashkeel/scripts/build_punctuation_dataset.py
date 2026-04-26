from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from punct_common import (
    LABELS,
    copy_if_needed,
    default_ml_root,
    default_project_root,
    extract_words_and_labels,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Build ML punctuation restoration datasets.")
    parser.add_argument(
        "--train-source",
        default=os.path.join(
            project_root,
            "datasets",
            "arabic_tashkeel",
            "splits",
            "final_tashkeel_train.jsonl",
        ),
    )
    parser.add_argument(
        "--eval-source",
        default=os.path.join(
            project_root,
            "datasets",
            "arabic_tashkeel",
            "splits",
            "final_tashkeel_eval.jsonl",
        ),
    )
    parser.add_argument("--source-copy-dir", default=os.path.join(ml_root, "data", "source"))
    parser.add_argument("--out-dir", default=os.path.join(ml_root, "data", "processed"))
    parser.add_argument("--reports-dir", default=os.path.join(ml_root, "reports"))
    parser.add_argument("--train-output-name", default="punctuation_train.jsonl")
    parser.add_argument("--eval-output-name", default="punctuation_eval.jsonl")
    parser.add_argument("--stats-output-name", default="dataset_stats.json")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit. 0 means all rows.")
    return parser.parse_args()


def build_split(src_path: str, dst_path: str, split: str, min_words: int, max_rows: int = 0) -> dict:
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    skipped = 0
    scanned = 0

    def rows():
        nonlocal skipped, scanned
        for row in read_jsonl(src_path):
            if max_rows and scanned >= max_rows:
                break
            scanned += 1
            source = str(row.get("source", "unknown"))
            text = str(row.get("assistant") or row.get("text") or "")
            example = extract_words_and_labels(text, source=source)
            if example is None or len(example.words) < min_words:
                skipped += 1
                continue
            label_counts.update(example.labels)
            source_counts[source] += 1
            yield {
                "split": split,
                "source": example.source,
                "words": example.words,
                "labels": example.labels,
                "text_unpunctuated": example.text_unpunctuated,
                "text_punctuated": example.text_punctuated,
            }

    written = write_jsonl(dst_path, rows())
    return {
        "source_path": src_path,
        "output_path": dst_path,
        "scanned_rows": scanned,
        "written_rows": written,
        "skipped_rows": skipped,
        "label_counts": {label: int(label_counts.get(label, 0)) for label in LABELS},
        "source_counts": dict(source_counts.most_common()),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.source_copy_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    copied_train = os.path.join(args.source_copy_dir, os.path.basename(args.train_source))
    copied_eval = os.path.join(args.source_copy_dir, os.path.basename(args.eval_source))
    copy_if_needed(args.train_source, copied_train)
    copy_if_needed(args.eval_source, copied_eval)

    train_out = os.path.join(args.out_dir, args.train_output_name)
    eval_out = os.path.join(args.out_dir, args.eval_output_name)

    train_stats = build_split(copied_train, train_out, "train", args.min_words, args.max_rows)
    eval_stats = build_split(copied_eval, eval_out, "eval", args.min_words, args.max_rows)

    stats = {
        "task": "word-boundary punctuation restoration",
        "labels": LABELS,
        "train": train_stats,
        "eval": eval_stats,
    }

    stats_path = os.path.join(args.reports_dir, args.stats_output_name)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print("Punctuation dataset built.")
    print(f"Copied train source: {copied_train}")
    print(f"Copied eval source:  {copied_eval}")
    print(f"Train output:        {train_out}")
    print(f"Eval output:         {eval_out}")
    print(f"Stats:               {stats_path}")
    print()
    print("Train rows:", train_stats["written_rows"])
    print("Eval rows: ", eval_stats["written_rows"])
    print("Train label counts:")
    for label in LABELS:
        print(f"  {label:12s} {train_stats['label_counts'][label]}")


if __name__ == "__main__":
    main()
