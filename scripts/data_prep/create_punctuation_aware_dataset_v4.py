"""Clone the current canonical JSONL splits into punctuation-aware v4 copies.

Why this exists:
  - The current canonical JSONLs already contain punctuation in many assistant
    rows, but the active CATT builders strip it out before training.
  - v4 keeps the raw split content intact under new filenames so the
    punctuation-aware training pipeline has an explicit, versioned source.

Output:
  datasets/arabic_tashkeel/splits/final_tashkeel_train_punct_v4.jsonl
  datasets/arabic_tashkeel/splits/final_tashkeel_eval_punct_v4.jsonl
  datasets/arabic_tashkeel/splits/final_tashkeel_punct_v4_report.txt
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLITS_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits")

SRC_TRAIN = os.path.join(SPLITS_DIR, "final_tashkeel_train.jsonl")
SRC_EVAL = os.path.join(SPLITS_DIR, "final_tashkeel_eval.jsonl")

OUT_TRAIN = os.path.join(SPLITS_DIR, "final_tashkeel_train_punct_v4.jsonl")
OUT_EVAL = os.path.join(SPLITS_DIR, "final_tashkeel_eval_punct_v4.jsonl")
OUT_REPORT = os.path.join(SPLITS_DIR, "final_tashkeel_punct_v4_report.txt")

PUNCT_CHARS = '،؛:,.!?؟()[]«»"()-'
PUNCT_RE = re.compile(f"[{re.escape(PUNCT_CHARS)}]")


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def punctuation_count(text: str) -> int:
    return len(PUNCT_RE.findall(text or ""))


def clone_split(src_path: str, dst_path: str):
    rows = list(iter_jsonl(src_path))
    stats = {
        "rows": len(rows),
        "assistant_with_punctuation": 0,
        "assistant_dense_punctuation": 0,
        "by_source": Counter(),
        "by_source_with_punctuation": Counter(),
        "by_source_dense_punctuation": Counter(),
    }

    with open(dst_path, "w", encoding="utf-8") as out:
        for row in rows:
            source = row.get("source", "unknown")
            assistant = row.get("assistant", "")
            punct = punctuation_count(assistant)

            stats["by_source"][source] += 1
            if punct > 0:
                stats["assistant_with_punctuation"] += 1
                stats["by_source_with_punctuation"][source] += 1
            if punct >= 2:
                stats["assistant_dense_punctuation"] += 1
                stats["by_source_dense_punctuation"][source] += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def write_report(train_stats: dict, eval_stats: dict) -> None:
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("Punctuation-aware dataset v4 report\n")
        handle.write("===================================\n\n")
        handle.write(f"Source train: {SRC_TRAIN}\n")
        handle.write(f"Source eval : {SRC_EVAL}\n\n")
        handle.write(f"Output train: {OUT_TRAIN}\n")
        handle.write(f"Output eval : {OUT_EVAL}\n\n")

        for label, stats in (("Train", train_stats), ("Eval", eval_stats)):
            handle.write(f"{label}\n")
            handle.write(f"{'-' * len(label)}\n")
            handle.write(f"Rows: {stats['rows']}\n")
            handle.write(f"Assistant rows with punctuation: {stats['assistant_with_punctuation']}\n")
            handle.write(f"Assistant rows with 2+ punctuation marks: {stats['assistant_dense_punctuation']}\n")
            handle.write("Top sources by rows:\n")
            for source, count in stats["by_source"].most_common(12):
                punct_rows = stats["by_source_with_punctuation"][source]
                dense_rows = stats["by_source_dense_punctuation"][source]
                handle.write(
                    f"  {source:<22} total={count:<6} punct={punct_rows:<6} dense={dense_rows}\n"
                )
            handle.write("\n")


def main() -> None:
    train_stats = clone_split(SRC_TRAIN, OUT_TRAIN)
    eval_stats = clone_split(SRC_EVAL, OUT_EVAL)
    write_report(train_stats, eval_stats)

    print("Created punctuation-aware v4 JSONL copies:")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_EVAL}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
