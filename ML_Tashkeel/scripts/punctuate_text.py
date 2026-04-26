from __future__ import annotations

import argparse
import os
import sys

from punct_common import (
    LABELS,
    LABEL_TO_OUTPUT,
    default_ml_root,
    extract_words_and_labels,
    punctuation_label_from_gap,
    reconstruct_text,
    strip_diacritics,
)
from punct_modeling import choose_device, load_model, load_tokenizer, predict_labels_for_words


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Restore punctuation in Arabic text.")
    parser.add_argument("--model-dir", default=os.path.join(ml_root, "models", "punctuation_restorer"))
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--none-threshold", type=float, default=0.0)
    parser.add_argument(
        "--label-thresholds",
        default="",
        help="Optional per-label thresholds, e.g. COMMA=0.95,COLON=0.9,SEMICOLON=0.98.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow model predictions to replace punctuation already present after a word.",
    )
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            return handle.read()
    return args.text


def parse_label_thresholds(value: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if not value.strip():
        return thresholds
    valid_labels = set(LABELS) - {"NONE"}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid threshold item {item!r}; expected LABEL=VALUE.")
        label, raw_threshold = item.split("=", 1)
        label = label.strip().upper()
        if label not in valid_labels:
            raise ValueError(f"Unknown punctuation label {label!r}; expected one of {sorted(valid_labels)}.")
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold for {label} must be between 0 and 1.")
        thresholds[label] = threshold
    return thresholds


def restore_punctuation(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    max_words: int = 180,
    none_threshold: float = 0.0,
    label_thresholds: dict[str, float] | None = None,
    preserve_existing: bool = True,
) -> tuple[str, list[str], list[float]]:
    example = extract_words_and_labels(strip_diacritics(text), source="input")
    if example is None:
        return text, [], []

    device = choose_device()
    predicted_labels, confidences = predict_labels_for_words(
        model=model,
        tokenizer=tokenizer,
        words=example.words,
        max_length=max_length,
        max_words=max_words,
        none_threshold=none_threshold,
        label_thresholds=label_thresholds,
        device=device,
    )

    final_labels = predicted_labels
    if preserve_existing:
        final_labels = []
        for existing, predicted in zip(example.labels, predicted_labels):
            final_labels.append(existing if existing != "NONE" else predicted)

    return reconstruct_text(example.words, final_labels), final_labels, confidences


def main() -> None:
    ensure_utf8_console()
    args = parse_args()
    text = load_input_text(args)
    if not text.strip():
        raise ValueError("Provide --text or --file with Arabic input.")

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir)
    output, labels, confidences = restore_punctuation(
        text=text,
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_words=args.max_words,
        none_threshold=args.none_threshold,
        label_thresholds=parse_label_thresholds(args.label_thresholds),
        preserve_existing=not args.overwrite_existing,
    )

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(output)

    print(output)
    print()
    print("Telemetry")
    print(f"  model_dir:          {args.model_dir}")
    print(f"  word_boundaries:    {len(labels)}")
    print(f"  predicted_punct:    {sum(1 for label in labels if LABEL_TO_OUTPUT.get(label))}")
    if confidences:
        print(f"  mean_confidence:    {sum(confidences) / len(confidences):.4f}")


if __name__ == "__main__":
    main()
