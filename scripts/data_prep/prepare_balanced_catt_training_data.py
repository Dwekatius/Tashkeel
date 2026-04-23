"""Build a harder, larger CATT-format training set from the canonical JSONLs.

Strategy:
  - Keep every original training row.
  - Upsample harder sources that still lag in eval:
      * parquet:shamela   -> 2x total copies
      * parquet:wikipedia -> 6x total copies
      * quran             -> 7x total copies
  - Keep every original long_passages_v1 row and also add sentence-aware
    chunks so long-form prose contributes more usable training rows without
    relying only on very long sequences.
  - Leave eval/test unchanged so validation stays apples-to-apples.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_TRAIN = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_train.jsonl")
SRC_EVAL = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_eval.jsonl")
OUT_ROOT = os.path.join(ROOT, "datasets", "arabic_tashkeel", "catt_format_balanced_v1")

TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")
KEEP_RE = re.compile(r"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]")
STRONG_SPLIT_RE = re.compile(r"(?<=[\.\u061f\u061b])\s+")

SOURCE_MULTIPLIERS = {
    "parquet:shamela": 2,
    "parquet:wikipedia": 6,
    "quran": 7,
}

LONG_SOURCE = "long_passages_v1"
LONG_CHUNK_MIN_LEN = 140
LONG_CHUNK_TARGET_LEN = 320
LONG_CHUNK_MAX_LEN = 420
LONG_CHUNK_OVERLAP = 1


def clean_for_catt(text: str) -> str:
    text = text.replace("ٱ", "ا")
    text = text.replace("\u0640", "")
    text = KEEP_RE.sub(" ", text)
    return " ".join(text.split())


def is_usable(text: str) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if not TASHKEEL_RE.search(text):
        return False, "no_tashkeel"
    if len(text) <= 5:
        return False, "short"
    return True, "ok"


def split_long_unit(unit: str, hard_max: int) -> list[str]:
    unit = unit.strip()
    if len(unit) <= hard_max:
        return [unit] if unit else []

    comma_parts = [part.strip() for part in re.split(r"(?<=،)\s+", unit) if part.strip()]
    if len(comma_parts) > 1:
        pieces = []
        current = ""
        for part in comma_parts:
            candidate = part if not current else f"{current} {part}"
            if current and len(candidate) > hard_max:
                pieces.append(current)
                current = part
            else:
                current = candidate
        if current:
            pieces.append(current)
        if all(len(piece) <= hard_max for piece in pieces):
            return pieces

    words = unit.split()
    pieces = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word])
        if current_words and len(candidate) > hard_max:
            pieces.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)
    if current_words:
        pieces.append(" ".join(current_words))
    return [piece for piece in pieces if piece]


def split_into_units(text: str) -> list[str]:
    units = []
    for unit in STRONG_SPLIT_RE.split(text):
        unit = unit.strip()
        if not unit:
            continue
        units.extend(split_long_unit(unit, LONG_CHUNK_MAX_LEN))
    return units


def chunk_long_passage(text: str) -> list[str]:
    units = split_into_units(text)
    if len(units) <= 1:
        return []

    chunks = []
    start = 0
    while start < len(units):
        current = units[start]
        end = start + 1

        while end < len(units):
            candidate = f"{current} {units[end]}"
            if len(candidate) > LONG_CHUNK_MAX_LEN:
                break
            if len(current) >= LONG_CHUNK_TARGET_LEN:
                break
            current = candidate
            end += 1

        if len(current) >= LONG_CHUNK_MIN_LEN and current != text:
            chunks.append(current)

        if end >= len(units):
            break

        next_start = max(start + 1, end - LONG_CHUNK_OVERLAP)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    deduped = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            deduped.append(chunk)
            seen.add(chunk)
    return deduped


def write_plain_split(jsonl_path: str, out_txt: str) -> tuple[int, Counter]:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    kept = 0
    skipped = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(out_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cleaned = clean_for_catt(row.get("assistant", ""))
            usable, reason = is_usable(cleaned)
            if not usable:
                skipped[reason] += 1
                continue
            fout.write(cleaned + "\n")
            kept += 1
    return kept, skipped


def write_balanced_train(jsonl_path: str, out_txt: str) -> dict:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    stats = {
        "base_rows": 0,
        "written_rows": 0,
        "skipped": Counter(),
        "written_by_source": Counter(),
        "extra_duplicate_rows_by_source": Counter(),
        "extra_chunk_rows_by_source": Counter(),
        "long_chunk_rows": 0,
    }

    with open(jsonl_path, "r", encoding="utf-8") as fin, open(out_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row.get("source", "unknown")
            cleaned = clean_for_catt(row.get("assistant", ""))
            usable, reason = is_usable(cleaned)
            if not usable:
                stats["skipped"][reason] += 1
                continue

            stats["base_rows"] += 1
            copies = SOURCE_MULTIPLIERS.get(source, 1)
            for copy_idx in range(copies):
                fout.write(cleaned + "\n")
                stats["written_rows"] += 1
                stats["written_by_source"][source] += 1
                if copy_idx > 0:
                    stats["extra_duplicate_rows_by_source"][source] += 1

            if source == LONG_SOURCE:
                for chunk in chunk_long_passage(cleaned):
                    usable_chunk, _ = is_usable(chunk)
                    if not usable_chunk:
                        continue
                    fout.write(chunk + "\n")
                    stats["written_rows"] += 1
                    stats["written_by_source"][source] += 1
                    stats["extra_chunk_rows_by_source"][source] += 1
                    stats["long_chunk_rows"] += 1

    return stats


def write_build_info(out_path: str, train_stats: dict, val_kept: int, val_skipped: Counter) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Balanced CATT dataset build info\n")
        handle.write("================================\n\n")
        handle.write(f"Output root: {OUT_ROOT}\n")
        handle.write(f"Train base usable rows: {train_stats['base_rows']}\n")
        handle.write(f"Train written rows: {train_stats['written_rows']}\n")
        handle.write(f"Long chunk rows added: {train_stats['long_chunk_rows']}\n")
        handle.write(f"Val rows kept: {val_kept}\n\n")
        handle.write("Source multipliers (total copies kept per original row)\n")
        for source, mult in SOURCE_MULTIPLIERS.items():
            handle.write(f"  {source}: {mult}x\n")
        handle.write("\nTrain rows written by source\n")
        for source, count in train_stats["written_by_source"].most_common():
            handle.write(f"  {source}: {count}\n")
        handle.write("\nExtra duplicate rows by source\n")
        for source, count in train_stats["extra_duplicate_rows_by_source"].most_common():
            handle.write(f"  {source}: {count}\n")
        handle.write("\nExtra chunk rows by source\n")
        for source, count in train_stats["extra_chunk_rows_by_source"].most_common():
            handle.write(f"  {source}: {count}\n")
        handle.write("\nSkipped train rows\n")
        for reason, count in train_stats["skipped"].items():
            handle.write(f"  {reason}: {count}\n")
        handle.write("\nSkipped val rows\n")
        for reason, count in val_skipped.items():
            handle.write(f"  {reason}: {count}\n")


def main():
    train_txt = os.path.join(OUT_ROOT, "train", "train.txt")
    val_txt = os.path.join(OUT_ROOT, "val", "val.txt")
    test_txt = os.path.join(OUT_ROOT, "test", "test.txt")
    info_txt = os.path.join(OUT_ROOT, "BUILD_INFO.txt")

    print(f"Writing balanced train split to {train_txt} ...")
    train_stats = write_balanced_train(SRC_TRAIN, train_txt)
    print(f"  base_usable={train_stats['base_rows']}  written={train_stats['written_rows']}  long_chunks={train_stats['long_chunk_rows']}")
    print("  top train sources after balancing:")
    for source, count in train_stats["written_by_source"].most_common(8):
        print(f"    {source:<22} {count}")

    print(f"Writing val split to {val_txt} ...")
    val_kept, val_skipped = write_plain_split(SRC_EVAL, val_txt)
    print(f"  kept={val_kept}")

    print(f"Writing {test_txt} (mirror of val) ...")
    with open(val_txt, "r", encoding="utf-8") as source_handle:
        val_data = source_handle.read()
    os.makedirs(os.path.dirname(test_txt), exist_ok=True)
    with open(test_txt, "w", encoding="utf-8") as test_handle:
        test_handle.write(val_data)
    print(f"  kept={val_kept}")

    write_build_info(info_txt, train_stats, val_kept, val_skipped)
    print(f"\nDone. Output root: {OUT_ROOT}")
    print(f"Build info: {info_txt}")


if __name__ == "__main__":
    main()
