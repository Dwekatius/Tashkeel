from __future__ import annotations

import argparse
import os

from punct_common import (
    LABEL_TO_OUTPUT,
    default_ml_root,
    extract_words_and_labels,
    punctuation_label_from_gap,
    reconstruct_text,
    strip_diacritics,
)
from punct_modeling import choose_device, load_model, load_tokenizer, predict_labels_for_words


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


def restore_punctuation(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    max_words: int = 180,
    none_threshold: float = 0.0,
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
        device=device,
    )

    final_labels = predicted_labels
    if preserve_existing:
        final_labels = []
        for existing, predicted in zip(example.labels, predicted_labels):
            final_labels.append(existing if existing != "NONE" else predicted)

    return reconstruct_text(example.words, final_labels), final_labels, confidences


def main() -> None:
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

