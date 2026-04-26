from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from copy import deepcopy
from typing import Callable

from punct_common import LABELS, default_ml_root, read_jsonl, reconstruct_text, write_jsonl


MODERN_SOURCES = {
    "easyarabicnews_v1",
    "long_passages_v1",
    "parquet:wikipedia",
    "ambiguous_bonus",
    "ambiguous_tier1",
    "ambiguous_tier2",
    "ambiguous_tier3",
    "ambiguous_tier4",
    "ambiguous_tier5",
    "ambiguous_tier6",
    "ambiguous_tier7",
    "ambiguous_tier8",
    "ambiguous_tier9",
    "ambiguous_tier10",
    "ambiguous_tier11",
}

QUESTION_STARTERS = {
    "هل",
    "لماذا",
    "كيف",
    "متى",
    "أين",
    "أينما",
    "ماذا",
    "أليس",
    "أليست",
    "ألا",
}

EXCLAMATION_STARTERS = {
    "يا",
    "أيها",
    "أيتها",
    "هيا",
    "دعونا",
    "فلن",
    "لن",
    "انتبه",
    "انتبهوا",
    "احذر",
    "احذروا",
    "تخيل",
    "تخيلوا",
}

EXCLAMATION_CUES = {
    "حقا",
    "حقاً",
    "جدا",
    "جداً",
    "للغاية",
    "بالتأكيد",
    "رائع",
    "رائعة",
    "ممتاز",
    "ممتازة",
    "عظيم",
    "عظيمة",
    "مذهل",
    "مذهلة",
    "استثنائي",
    "استثنائية",
}

SPEECH_COLON_CUES = {
    "قال",
    "قالت",
    "قالوا",
    "قلت",
    "يقول",
    "تقول",
    "سأل",
    "سألت",
    "يسأل",
    "تسأل",
    "أجاب",
    "أجابت",
    "أضاف",
    "أضافت",
}

LIST_COLON_CUES = {
    "التالي",
    "التالية",
    "هما",
    "وهي",
    "مثل",
    "أبرزها",
    "أهمها",
}

LONG_SPEAKER_PHRASE_STARTS = {
    "وزارة",
    "وكالة",
    "صحيفة",
    "جامعة",
    "شركة",
    "الحكومة",
    "السلطات",
    "الشرطة",
    "المؤسسة",
    "الجمعية",
    "اللجنة",
}

SPEECH_CUE_FOLLOWER_STOPWORDS = {
    "إن",
    "إنه",
    "إنها",
    "أن",
    "أنه",
    "أنها",
    "عن",
    "عنه",
    "عنها",
    "في",
    "على",
    "من",
    "ل",
    "له",
    "لها",
    "ذلك",
    "هذا",
    "هذه",
}

TWO_WORD_SPEAKER_STARTS = {
    "متحدث",
    "رئيس",
    "مدير",
    "وزير",
    "قائد",
    "مسؤول",
    "خبير",
}


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Build targeted v2 augmentation for punctuation training.")
    parser.add_argument("--train-file", default=os.path.join(ml_root, "data", "processed", "punctuation_train.jsonl"))
    parser.add_argument(
        "--output-file",
        default=os.path.join(ml_root, "data", "processed", "punctuation_train_aug_v2.jsonl"),
    )
    parser.add_argument("--reports-dir", default=os.path.join(ml_root, "reports"))
    parser.add_argument("--report-name", default="augmentation_v2_report.json")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--max-semicolon-rows", type=int, default=24000)
    parser.add_argument("--max-colon-rows", type=int, default=18000)
    parser.add_argument("--max-question-rows", type=int, default=12000)
    parser.add_argument("--max-exclamation-rows", type=int, default=14000)
    parser.add_argument("--max-exclamation-oversample", type=int, default=4000)
    parser.add_argument("--max-total-augmented-rows", type=int, default=70000)
    return parser.parse_args()


def label_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(row["labels"])
    return counts


def count_rows_with_labels(rows: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        present = set(row["labels"])
        for label in LABELS:
            if label in present:
                counts[label] += 1
    return {label: int(counts.get(label, 0)) for label in LABELS}


def sentence_spans(labels: list[str]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for idx, label in enumerate(labels):
        if label != "NONE":
            spans.append((start, idx))
            start = idx + 1
    if start < len(labels):
        spans.append((start, len(labels) - 1))
    return [(start, end) for start, end in spans if start <= end]


def previous_boundary_distance(labels: list[str], index: int) -> int:
    distance = 0
    for idx in range(index - 1, -1, -1):
        distance += 1
        if labels[idx] != "NONE":
            break
    return distance


def next_boundary_distance(labels: list[str], index: int) -> int:
    distance = 0
    for idx in range(index + 1, len(labels)):
        distance += 1
        if labels[idx] != "NONE":
            break
    return distance


def is_modern(row: dict) -> bool:
    return row.get("source") in MODERN_SOURCES


def make_augmented_row(row: dict, labels: list[str], rule: str) -> dict:
    new_row = deepcopy(row)
    new_row["labels"] = labels
    new_row["text_punctuated"] = reconstruct_text(new_row["words"], labels)
    new_row["source"] = f"{row.get('source', 'unknown')}|aug_v2:{rule}"
    new_row["augmentation"] = {
        "version": "aug_v2",
        "rule": rule,
        "original_source": row.get("source", "unknown"),
    }
    return new_row


def semicolon_variant(row: dict) -> dict | None:
    labels = row["labels"]
    candidates = []
    for idx, label in enumerate(labels):
        if label == "COMMA" and previous_boundary_distance(labels, idx) >= 7 and next_boundary_distance(labels, idx) >= 7:
            candidates.append(idx)
        elif label == "PERIOD" and idx < len(labels) - 1:
            if previous_boundary_distance(labels, idx) >= 8 and next_boundary_distance(labels, idx) >= 8:
                candidates.append(idx)
    if not candidates:
        return None
    new_labels = labels.copy()
    new_labels[candidates[0]] = "SEMICOLON"
    return make_augmented_row(row, new_labels, "semicolon_from_long_clause_boundary")


def colon_variant(row: dict) -> dict | None:
    words = row["words"]
    labels = row["labels"]
    for idx, word in enumerate(words[:-2]):
        follower = words[idx + 1]
        if follower in LONG_SPEAKER_PHRASE_STARTS or follower in SPEECH_CUE_FOLLOWER_STOPWORDS:
            continue
        if follower in SPEECH_COLON_CUES:
            continue
        target = idx + 2 if follower in TWO_WORD_SPEAKER_STARTS and idx + 2 < len(words) else idx + 1
        if word in SPEECH_COLON_CUES and labels[target] in {"NONE", "COMMA", "PERIOD"}:
            new_labels = labels.copy()
            new_labels[target] = "COLON"
            return make_augmented_row(row, new_labels, "colon_after_speech_or_list_cue")

    for idx, word in enumerate(words[:-1]):
        if row.get("source") != "parquet:wikipedia" and word in LIST_COLON_CUES and labels[idx] in {
            "NONE",
            "COMMA",
            "PERIOD",
        }:
            new_labels = labels.copy()
            new_labels[idx] = "COLON"
            return make_augmented_row(row, new_labels, "colon_after_speech_or_list_cue")
    return None


def question_variant(row: dict) -> dict | None:
    words = row["words"]
    labels = row["labels"]
    if len(words) > 20:
        return None
    for start, end in sentence_spans(labels):
        if end - start < 2:
            continue
        if words[start] in QUESTION_STARTERS and labels[end] in {"NONE", "PERIOD", "COMMA"}:
            new_labels = labels.copy()
            new_labels[end] = "QUESTION"
            return make_augmented_row(row, new_labels, "question_from_interrogative_sentence")
    return None


def exclamation_variant(row: dict) -> dict | None:
    words = row["words"]
    labels = row["labels"]
    word_set = set(words)
    for start, end in sentence_spans(labels):
        if end - start < 2:
            continue
        sentence_words = words[start : end + 1]
        sentence_word_set = set(sentence_words)
        starts_like_exclamation = sentence_words[0] in EXCLAMATION_STARTERS
        has_emphasis = bool(sentence_word_set & EXCLAMATION_CUES)
        if (starts_like_exclamation or has_emphasis) and labels[end] in {"NONE", "PERIOD", "COMMA", "QUESTION"}:
            new_labels = labels.copy()
            new_labels[end] = "EXCLAMATION"
            return make_augmented_row(row, new_labels, "exclamation_from_vocative_or_emphasis")

    if len(words) >= 8 and word_set & {"يجب", "ينبغي", "علينا", "لابد", "ضروري"}:
        for idx in range(len(labels) - 1, -1, -1):
            if labels[idx] in {"PERIOD", "COMMA"}:
                new_labels = labels.copy()
                new_labels[idx] = "EXCLAMATION"
                return make_augmented_row(row, new_labels, "exclamation_from_urgent_statement")
    return None


def oversample_existing_label(row: dict, label: str) -> dict | None:
    if label not in row["labels"]:
        return None
    return make_augmented_row(row, row["labels"].copy(), f"oversample_existing_{label.lower()}")


def add_variants(
    rows: list[dict],
    shuffled_indices: list[int],
    rule_name: str,
    limit: int,
    maker: Callable[[dict], dict | None],
    augmented: list[dict],
    rule_counts: Counter,
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]],
    max_total: int,
) -> None:
    if limit <= 0 or len(augmented) >= max_total:
        return
    added = 0
    for index in shuffled_indices:
        if added >= limit or len(augmented) >= max_total:
            break
        variant = maker(rows[index])
        if variant is None:
            continue
        key = (tuple(variant["words"]), tuple(variant["labels"]), rule_name)
        if key in seen:
            continue
        seen.add(key)
        augmented.append(variant)
        rule_counts[rule_name] += 1
        added += 1


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    rows = list(read_jsonl(args.train_file))
    original_counts = label_counts(rows)
    original_row_label_counts = count_rows_with_labels(rows)

    rng = random.Random(args.seed)
    all_indices = list(range(len(rows)))
    rng.shuffle(all_indices)
    modern_indices = [idx for idx in all_indices if is_modern(rows[idx])]

    augmented: list[dict] = []
    rule_counts: Counter = Counter()
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()

    add_variants(
        rows,
        modern_indices,
        "semicolon_from_long_clause_boundary",
        args.max_semicolon_rows,
        semicolon_variant,
        augmented,
        rule_counts,
        seen,
        args.max_total_augmented_rows,
    )
    add_variants(
        rows,
        modern_indices,
        "colon_after_speech_or_list_cue",
        args.max_colon_rows,
        colon_variant,
        augmented,
        rule_counts,
        seen,
        args.max_total_augmented_rows,
    )
    add_variants(
        rows,
        modern_indices,
        "question_from_interrogative_sentence",
        args.max_question_rows,
        question_variant,
        augmented,
        rule_counts,
        seen,
        args.max_total_augmented_rows,
    )
    add_variants(
        rows,
        modern_indices,
        "exclamation_from_vocative_or_emphasis",
        args.max_exclamation_rows,
        exclamation_variant,
        augmented,
        rule_counts,
        seen,
        args.max_total_augmented_rows,
    )
    add_variants(
        rows,
        all_indices,
        "oversample_existing_exclamation",
        args.max_exclamation_oversample,
        lambda row: oversample_existing_label(row, "EXCLAMATION"),
        augmented,
        rule_counts,
        seen,
        args.max_total_augmented_rows,
    )

    output_rows = rows + augmented
    augmented_counts = label_counts(augmented)
    output_counts = label_counts(output_rows)
    actual_rule_counts = Counter(row["augmentation"]["rule"] for row in augmented)

    written = write_jsonl(args.output_file, output_rows)

    report = {
        "version": "aug_v2",
        "train_file": args.train_file,
        "output_file": args.output_file,
        "seed": args.seed,
        "original_rows": len(rows),
        "augmented_rows": len(augmented),
        "written_rows": written,
        "rule_counts": {key: int(value) for key, value in actual_rule_counts.items()},
        "original_label_counts": {label: int(original_counts.get(label, 0)) for label in LABELS},
        "augmented_label_counts": {label: int(augmented_counts.get(label, 0)) for label in LABELS},
        "output_label_counts": {label: int(output_counts.get(label, 0)) for label in LABELS},
        "original_rows_with_label": original_row_label_counts,
        "output_rows_with_label": count_rows_with_labels(output_rows),
        "notes": [
            "The original processed training rows are kept unchanged.",
            "Augmented rows are tagged in source with |aug_v2:<rule>.",
            "Quote marks are not augmented because the current label set does not support quote restoration.",
            "punc_tests.txt is not used as training data; it remains a manual QA set.",
        ],
    }

    report_path = os.path.join(args.reports_dir, args.report_name)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("Augmented punctuation dataset built.")
    print(f"Original rows:  {len(rows)}")
    print(f"Augmented rows: {len(augmented)}")
    print(f"Written rows:   {written}")
    print(f"Output file:    {args.output_file}")
    print(f"Report:         {report_path}")
    print()
    print("Rule counts:")
    for rule, count in actual_rule_counts.most_common():
        print(f"  {rule:42s} {count}")
    print()
    print("Output label counts:")
    for label in LABELS:
        before = int(original_counts.get(label, 0))
        after = int(output_counts.get(label, 0))
        print(f"  {label:12s} {before:9d} -> {after:9d} ({after - before:+d})")


if __name__ == "__main__":
    main()
