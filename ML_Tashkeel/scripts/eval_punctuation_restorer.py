from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification

from punct_common import LABELS, LABEL_TO_ID, default_ml_root, read_jsonl
from punct_modeling import (
    PunctuationTokenDataset,
    Timer,
    choose_device,
    compute_metrics_from_arrays,
    format_seconds,
    load_model,
    load_tokenizer,
    predict_labels_for_words,
)


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Evaluate Arabic punctuation restoration model.")
    parser.add_argument("--model-dir", default=os.path.join(ml_root, "models", "punctuation_restorer"))
    parser.add_argument("--eval-file", default=os.path.join(ml_root, "data", "processed", "punctuation_eval.jsonl"))
    parser.add_argument("--reports-dir", default=os.path.join(ml_root, "reports"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--sequence-exact-limit", type=int, default=2000)
    parser.add_argument("--progress-bar", action="store_true", help="Show tqdm progress bars.")
    return parser.parse_args()


@torch.inference_mode()
def collect_token_predictions(
    model,
    dataloader,
    device: torch.device,
    progress_bar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[int] = []
    all_predictions: list[int] = []
    model.to(device)
    model.eval()

    for batch in tqdm(dataloader, desc="Token eval", disable=not progress_bar):
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits.cpu(), dim=-1)
        labels_np = labels.numpy()
        predictions_np = predictions.numpy()
        mask = labels_np != -100
        all_labels.extend(labels_np[mask].astype(int).tolist())
        all_predictions.extend(predictions_np[mask].astype(int).tolist())

    return np.array(all_labels, dtype=np.int64), np.array(all_predictions, dtype=np.int64)


def compute_sequence_exact(
    model,
    tokenizer,
    eval_file: str,
    limit: int,
    device: torch.device,
    max_length: int,
    max_words: int,
    progress_bar: bool = False,
) -> dict:
    checked = 0
    exact = 0
    punct_exact = 0
    punct_rows = 0

    for row in tqdm(read_jsonl(eval_file), desc="Sequence exact", disable=not progress_bar):
        if limit and checked >= limit:
            break
        words = row["words"]
        labels = row["labels"]
        pred_labels, _ = predict_labels_for_words(
            model=model,
            tokenizer=tokenizer,
            words=words,
            max_length=max_length,
            max_words=max_words,
            batch_size=1,
            device=device,
        )
        checked += 1
        if pred_labels == labels:
            exact += 1
        gold_punct = [label for label in labels if label != "NONE"]
        pred_punct = [label for label in pred_labels if label != "NONE"]
        if gold_punct:
            punct_rows += 1
            if pred_punct == gold_punct:
                punct_exact += 1

    return {
        "checked_rows": checked,
        "exact_rows": exact,
        "exact_rate": exact / checked if checked else 0.0,
        "punct_rows": punct_rows,
        "punct_sequence_exact_rows": punct_exact,
        "punct_sequence_exact_rate": punct_exact / punct_rows if punct_rows else 0.0,
    }


def write_confusion_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def main() -> None:
    args = parse_args()
    timer = Timer()
    os.makedirs(args.reports_dir, exist_ok=True)

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file does not exist: {args.eval_file}")

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir)
    device = choose_device()

    dataset = PunctuationTokenDataset(
        args.eval_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_words=args.max_words,
        max_examples=args.max_eval_examples or None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
    )

    print("Arabic punctuation restoration eval")
    print(f"Model dir:     {args.model_dir}")
    print(f"Eval file:     {args.eval_file}")
    print(f"Device:        {device.type}")
    print(f"Rows/chunks:   {dataset.row_count}/{len(dataset)}")
    print(f"Batch size:    {args.batch_size}")
    print()

    labels, predictions = collect_token_predictions(model, dataloader, device, args.progress_bar)
    mask = labels != -100
    y_true = labels[mask]
    y_pred = predictions[mask]
    metrics = compute_metrics_from_arrays(labels, predictions)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    seq_metrics = compute_sequence_exact(
        model=model,
        tokenizer=tokenizer,
        eval_file=args.eval_file,
        limit=args.sequence_exact_limit,
        device=device,
        max_length=args.max_length,
        max_words=args.max_words,
        progress_bar=args.progress_bar,
    )

    output = {
        "model_dir": args.model_dir,
        "eval_file": args.eval_file,
        "device": device.type,
        "rows": dataset.row_count,
        "chunks": len(dataset),
        "token_metrics": metrics,
        "classification_report": report,
        "sequence_metrics": seq_metrics,
        "elapsed_seconds": timer.elapsed(),
    }

    report_path = os.path.join(args.reports_dir, "punctuation_eval_report.json")
    confusion_path = os.path.join(args.reports_dir, "punctuation_confusion_matrix.csv")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    write_confusion_csv(confusion_path, y_true, y_pred)

    print("Metrics")
    for key, value in metrics.items():
        print(f"  {key:22s} {value:.4f}")
    print("Sequence")
    for key, value in seq_metrics.items():
        if isinstance(value, float):
            print(f"  {key:22s} {value:.4f}")
        else:
            print(f"  {key:22s} {value}")
    print()
    print(f"Elapsed:          {format_seconds(timer.elapsed())}")
    print(f"Report:           {report_path}")
    print(f"Confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()
