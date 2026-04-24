from __future__ import annotations

import argparse
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
from sklearn.metrics import classification_report
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
    parser = argparse.ArgumentParser(description="Tune per-label punctuation confidence thresholds.")
    parser.add_argument("--model-dir", default=os.path.join(ml_root, "models", "punctuation_restorer"))
    parser.add_argument("--eval-file", default=os.path.join(ml_root, "data", "processed", "punctuation_eval.jsonl"))
    parser.add_argument("--reports-dir", default=os.path.join(ml_root, "reports"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--metric", default="punct_micro_f1")
    parser.add_argument(
        "--candidate-thresholds",
        default="0,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99",
    )
    parser.add_argument("--sequence-exact-limit", type=int, default=2000)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def parse_threshold_values(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Candidate thresholds must be between 0 and 1.")
    if 0.0 not in thresholds:
        thresholds.insert(0, 0.0)
    return thresholds


@torch.inference_mode()
def collect_token_probabilities(
    model,
    dataloader,
    device: torch.device,
    progress_bar: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    model.to(device)
    model.eval()

    for batch in tqdm(dataloader, desc="Collect probs", disable=not progress_bar):
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        labels_np = labels.numpy()
        mask = labels_np != -100
        all_labels.extend(labels_np[mask].astype(int).tolist())
        all_probs.extend(probs[mask])

    return np.array(all_labels, dtype=np.int64), np.array(all_probs, dtype=np.float32)


def predict_with_thresholds(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    predictions = np.argmax(probs, axis=-1).astype(np.int64)
    for label, threshold in thresholds.items():
        if label == "NONE" or threshold <= 0.0:
            continue
        label_id = LABEL_TO_ID[label]
        demote = (predictions == label_id) & (probs[:, label_id] < threshold)
        predictions[demote] = LABEL_TO_ID["NONE"]
    return predictions


def score_thresholds(y_true: np.ndarray, probs: np.ndarray, thresholds: dict[str, float]) -> dict[str, float]:
    predictions = predict_with_thresholds(probs, thresholds)
    return compute_metrics_from_arrays(y_true, predictions)


def tune_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    candidate_thresholds: list[float],
    metric_name: str,
    passes: int,
) -> tuple[dict[str, float], dict[str, float]]:
    punctuation_labels = [label for label in LABELS if label != "NONE"]
    thresholds = {label: 0.0 for label in punctuation_labels}
    best_metrics = score_thresholds(y_true, probs, thresholds)
    if metric_name not in best_metrics:
        raise ValueError(f"Unknown metric {metric_name!r}; choose one of {sorted(best_metrics)}.")

    for _ in range(max(1, passes)):
        changed = False
        for label in punctuation_labels:
            best_label_threshold = thresholds[label]
            best_label_metrics = best_metrics
            for candidate in candidate_thresholds:
                trial = dict(thresholds)
                trial[label] = candidate
                metrics = score_thresholds(y_true, probs, trial)
                current_score = metrics[metric_name]
                best_score = best_label_metrics[metric_name]
                if current_score > best_score or (
                    current_score == best_score and candidate > best_label_threshold
                ):
                    best_label_threshold = candidate
                    best_label_metrics = metrics

            if best_label_threshold != thresholds[label]:
                thresholds[label] = best_label_threshold
                best_metrics = best_label_metrics
                changed = True

        if not changed:
            break

    return thresholds, best_metrics


def compute_sequence_exact(
    model,
    tokenizer,
    eval_file: str,
    limit: int,
    device: torch.device,
    max_length: int,
    max_words: int,
    label_thresholds: dict[str, float],
    progress_bar: bool,
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
            label_thresholds=label_thresholds,
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


def thresholds_for_cli(thresholds: dict[str, float]) -> str:
    return ",".join(f"{label}={value:g}" for label, value in thresholds.items() if value > 0.0)


def safe_model_name(model_dir: str) -> str:
    return os.path.basename(os.path.normpath(model_dir)) or "model"


def main() -> None:
    args = parse_args()
    timer = Timer()
    os.makedirs(args.reports_dir, exist_ok=True)

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

    print("Punctuation threshold tuning")
    print(f"Model dir:     {args.model_dir}")
    print(f"Eval file:     {args.eval_file}")
    print(f"Device:        {device.type}")
    print(f"Rows/chunks:   {dataset.row_count}/{len(dataset)}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Metric:        {args.metric}")
    print()

    y_true, probs = collect_token_probabilities(model, dataloader, device, args.progress_bar)
    candidates = parse_threshold_values(args.candidate_thresholds)
    baseline_thresholds = {label: 0.0 for label in LABELS if label != "NONE"}
    baseline_predictions = predict_with_thresholds(probs, baseline_thresholds)
    baseline_metrics = compute_metrics_from_arrays(y_true, baseline_predictions)
    baseline_report = classification_report(
        y_true,
        baseline_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    best_thresholds, best_metrics = tune_thresholds(
        y_true=y_true,
        probs=probs,
        candidate_thresholds=candidates,
        metric_name=args.metric,
        passes=args.passes,
    )
    tuned_predictions = predict_with_thresholds(probs, best_thresholds)
    tuned_report = classification_report(
        y_true,
        tuned_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    baseline_sequence = compute_sequence_exact(
        model=model,
        tokenizer=tokenizer,
        eval_file=args.eval_file,
        limit=args.sequence_exact_limit,
        device=device,
        max_length=args.max_length,
        max_words=args.max_words,
        label_thresholds=baseline_thresholds,
        progress_bar=args.progress_bar,
    )
    tuned_sequence = compute_sequence_exact(
        model=model,
        tokenizer=tokenizer,
        eval_file=args.eval_file,
        limit=args.sequence_exact_limit,
        device=device,
        max_length=args.max_length,
        max_words=args.max_words,
        label_thresholds=best_thresholds,
        progress_bar=args.progress_bar,
    )

    output = {
        "model_dir": args.model_dir,
        "eval_file": args.eval_file,
        "device": device.type,
        "rows": dataset.row_count,
        "chunks": len(dataset),
        "candidate_thresholds": candidates,
        "optimized_metric": args.metric,
        "baseline_metrics": baseline_metrics,
        "baseline_sequence_metrics": baseline_sequence,
        "baseline_classification_report": baseline_report,
        "best_thresholds": best_thresholds,
        "best_thresholds_cli": thresholds_for_cli(best_thresholds),
        "tuned_metrics": best_metrics,
        "tuned_sequence_metrics": tuned_sequence,
        "tuned_classification_report": tuned_report,
        "elapsed_seconds": timer.elapsed(),
    }

    report_path = os.path.join(args.reports_dir, f"punctuation_threshold_tuning_{safe_model_name(args.model_dir)}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print("Baseline")
    for key, value in baseline_metrics.items():
        print(f"  {key:22s} {value:.4f}")
    print("Tuned")
    for key, value in best_metrics.items():
        print(f"  {key:22s} {value:.4f}")
    print("Thresholds")
    for label, value in best_thresholds.items():
        print(f"  {label:12s} {value:.2f}")
    print("Sequence")
    print(f"  baseline_exact_rate      {baseline_sequence['exact_rate']:.4f}")
    print(f"  tuned_exact_rate         {tuned_sequence['exact_rate']:.4f}")
    print(f"  baseline_punct_seq_rate  {baseline_sequence['punct_sequence_exact_rate']:.4f}")
    print(f"  tuned_punct_seq_rate     {tuned_sequence['punct_sequence_exact_rate']:.4f}")
    print()
    print(f"CLI thresholds: {thresholds_for_cli(best_thresholds)}")
    print(f"Elapsed:        {format_seconds(timer.elapsed())}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    main()
