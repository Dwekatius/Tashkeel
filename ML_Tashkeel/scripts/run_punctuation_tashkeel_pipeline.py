from __future__ import annotations

import argparse
import os
import sys
import time

from punct_modeling import load_model, load_tokenizer
from punctuate_text import parse_label_thresholds, restore_punctuation


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_ml_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Run punctuation restoration then CATT v4 tashkeel.")
    parser.add_argument("--punct-model-dir", default=os.path.join(ml_root, "models", "punctuation_restorer"))
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
    parser.add_argument("--tashkeel-batch-size", type=int, default=32)
    parser.add_argument("--no-tashkeel", action="store_true")
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            return handle.read()
    return args.text


def main() -> None:
    args = parse_args()
    text = load_input_text(args)
    if not text.strip():
        raise ValueError("Provide --text or --file with Arabic input.")

    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    t0 = time.time()
    tokenizer = load_tokenizer(args.punct_model_dir)
    punct_model = load_model(args.punct_model_dir)
    punctuated, labels, confidences = restore_punctuation(
        text=text,
        model=punct_model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_words=args.max_words,
        none_threshold=args.none_threshold,
        label_thresholds=parse_label_thresholds(args.label_thresholds),
        preserve_existing=True,
    )
    punct_elapsed = time.time() - t0

    final_output = punctuated
    tashkeel_elapsed = 0.0
    checkpoint = ""
    device = ""

    if not args.no_tashkeel:
        from debug_tools.tashkeel_web_ui import TashkeelWebApp

        t1 = time.time()
        app = TashkeelWebApp(batch_size=args.tashkeel_batch_size)
        result = app.diacritize(punctuated)
        tashkeel_elapsed = time.time() - t1
        final_output = result["output"]
        checkpoint = result["checkpoint"]
        device = result["device"]

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(final_output)

    print(final_output)
    print()
    print("Pipeline telemetry")
    print(f"  punct_model_dir:       {args.punct_model_dir}")
    print(f"  punct_elapsed_sec:     {punct_elapsed:.3f}")
    print(f"  word_boundaries:       {len(labels)}")
    print(f"  predicted_punct:       {sum(1 for label in labels if label != 'NONE')}")
    if confidences:
        print(f"  mean_punct_confidence: {sum(confidences) / len(confidences):.4f}")
    if not args.no_tashkeel:
        print(f"  tashkeel_elapsed_sec:  {tashkeel_elapsed:.3f}")
        print(f"  tashkeel_device:       {device}")
        print(f"  tashkeel_checkpoint:   {checkpoint}")


if __name__ == "__main__":
    main()
