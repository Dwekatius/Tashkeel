from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from copy import deepcopy
from typing import Iterable

from punct_common import LABELS, default_ml_root, read_jsonl, write_jsonl


TARGET_LABELS = ["QUESTION", "COLON", "SEMICOLON", "EXCLAMATION"]


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(
        description="Build a real-data-only balanced punctuation training set with own_dataset_v1 emphasis."
    )
    parser.add_argument("--base-train-file", default=os.path.join(ml_root, "data", "processed", "punctuation_train.jsonl"))
    parser.add_argument(
        "--own-train-file",
        default=os.path.join(ml_root, "data", "processed", "punctuation_train_own_v1.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        default=os.path.join(ml_root, "data", "processed", "punctuation_train_own_v2_balanced.jsonl"),
    )
    parser.add_argument("--reports-dir", default=os.path.join(ml_root, "reports"))
    parser.add_argument("--report-name", default="balanced_own_v2_report.json")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--own-multiplier", type=int, default=6)
    parser.add_argument("--target-question", type=int, default=6500)
    parser.add_argument("--target-colon", type=int, default=42000)
    parser.add_argument("--target-semicolon", type=int, default=25000)
    parser.add_argument("--target-exclamation", type=int, default=2500)
    parser.add_argument("--max-total-rows", type=int, default=200000)
    return parser.parse_args()


def label_counts(rows: Iterable[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(row["labels"])
    return counts


def rows_with_label_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        for label in set(row["labels"]):
            counts[label] += 1
    return {label: int(counts.get(label, 0)) for label in LABELS}


def is_own_row(row: dict) -> bool:
    return str(row.get("source", "")) == "own_dataset_v1"


def clone_for_balance(row: dict, reason: str, copy_index: int) -> dict:
    cloned = deepcopy(row)
    cloned["source"] = f"{row.get('source', 'unknown')}|balanced_v2:{reason}"
    cloned["balance"] = {
        "version": "own_v2_balanced",
        "reason": reason,
        "copy_index": copy_index,
        "original_source": row.get("source", "unknown"),
    }
    return cloned


def cycle_candidates(candidates: list[dict], rng: random.Random) -> Iterable[dict]:
    if not candidates:
        return
    cycle = 0
    while True:
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        for row in shuffled:
            yield row, cycle
        cycle += 1


def add_until_label_target(
    output_rows: list[dict],
    candidates: list[dict],
    counts: Counter,
    label: str,
    target: int,
    reason: str,
    rng: random.Random,
    max_total_rows: int,
) -> int:
    if target <= 0 or counts.get(label, 0) >= target or not candidates:
        return 0

    added = 0
    for row, cycle in cycle_candidates(candidates, rng):
        if len(output_rows) >= max_total_rows or counts.get(label, 0) >= target:
            break
        cloned = clone_for_balance(row, reason, added + 1 + cycle * len(candidates))
        output_rows.append(cloned)
        counts.update(cloned["labels"])
        added += 1
    return added


def make_label_candidates(base_rows: list[dict], own_rows: list[dict], label: str) -> list[dict]:
    # Own examples are intentionally repeated first because they are manually
    # supplied, punctuation-dense, and much smaller than the base corpus.
    own = [row for row in own_rows if label in row["labels"]]
    base = [row for row in base_rows if label in row["labels"]]
    return own + base


def main() -> None:
    args = parse_args()
    if args.own_multiplier < 1:
        raise ValueError("--own-multiplier must be at least 1.")

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    base_rows = list(read_jsonl(args.base_train_file))
    combined_own_rows = list(read_jsonl(args.own_train_file))
    own_rows = [row for row in combined_own_rows if is_own_row(row)]

    output_rows = list(base_rows)
    additions: Counter = Counter()

    for copy_idx in range(args.own_multiplier):
        for row in own_rows:
            output_rows.append(clone_for_balance(row, "own_dataset_v1_multiplier", copy_idx + 1))
            additions["own_dataset_v1_multiplier"] += 1

    counts = label_counts(output_rows)
    targets = {
        "QUESTION": args.target_question,
        "COLON": args.target_colon,
        "SEMICOLON": args.target_semicolon,
        "EXCLAMATION": args.target_exclamation,
    }

    candidate_counts = {}
    for label in TARGET_LABELS:
        candidates = make_label_candidates(base_rows, own_rows, label)
        candidate_counts[label] = len(candidates)
        added = add_until_label_target(
            output_rows=output_rows,
            candidates=candidates,
            counts=counts,
            label=label,
            target=targets[label],
            reason=f"real_{label.lower()}_target",
            rng=rng,
            max_total_rows=args.max_total_rows,
        )
        additions[f"real_{label.lower()}_target"] = added

    written = write_jsonl(args.output_file, output_rows)
    output_counts = label_counts(output_rows)

    report = {
        "version": "own_v2_balanced",
        "base_train_file": args.base_train_file,
        "own_train_file": args.own_train_file,
        "output_file": args.output_file,
        "seed": args.seed,
        "policy": {
            "own_multiplier": args.own_multiplier,
            "targets": targets,
            "max_total_rows": args.max_total_rows,
            "notes": [
                "No punctuation labels are invented or changed.",
                "All added rows are duplicate copies of real rows that already contain their labels.",
                "own_dataset_v1 rows are repeated because the plain append was too small to affect training.",
            ],
        },
        "base_rows": len(base_rows),
        "own_rows_available": len(own_rows),
        "written_rows": written,
        "added_rows": int(written - len(base_rows)),
        "additions": {key: int(value) for key, value in additions.items()},
        "candidate_rows_by_label": candidate_counts,
        "base_label_counts": {label: int(label_counts(base_rows).get(label, 0)) for label in LABELS},
        "own_label_counts": {label: int(label_counts(own_rows).get(label, 0)) for label in LABELS},
        "output_label_counts": {label: int(output_counts.get(label, 0)) for label in LABELS},
        "output_rows_with_label": rows_with_label_counts(output_rows),
    }

    report_path = os.path.join(args.reports_dir, args.report_name)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("Balanced own punctuation dataset built.")
    print(f"Base rows:       {len(base_rows)}")
    print(f"Own rows:        {len(own_rows)}")
    print(f"Written rows:    {written}")
    print(f"Added rows:      {written - len(base_rows)}")
    print(f"Output file:     {args.output_file}")
    print(f"Report:          {report_path}")
    print()
    print("Additions:")
    for key, value in additions.items():
        print(f"  {key:30s} {int(value)}")
    print()
    print("Output label counts:")
    base_counts = label_counts(base_rows)
    for label in LABELS:
        before = int(base_counts.get(label, 0))
        after = int(output_counts.get(label, 0))
        print(f"  {label:12s} {before:9d} -> {after:9d} ({after - before:+d})")


if __name__ == "__main__":
    main()
