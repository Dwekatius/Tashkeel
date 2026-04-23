from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*", category=FutureWarning)

import torch
from torch import nn
from transformers import (
    DataCollatorForTokenClassification,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from tqdm.auto import tqdm

from punct_common import LABELS, LABEL_TO_ID, default_ml_root, read_jsonl
from punct_modeling import (
    PunctuationTokenDataset,
    Timer,
    format_seconds,
    load_model,
    load_tokenizer,
    save_label_files,
    trainer_compute_metrics,
)


DEFAULT_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"


class ProgressTelemetryCallback(TrainerCallback):
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.bar = None
        self.last_step = 0
        self.start_time = 0.0
        self.postfix: dict[str, str] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self.last_step = int(state.global_step)
        self.start_time = time.time()
        total = int(state.max_steps) if int(state.max_steps) > 0 else None
        self.bar = tqdm(
            total=total,
            initial=self.last_step,
            desc="training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            smoothing=0.05,
            file=sys.stdout,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if not self.bar:
            return
        step = int(state.global_step)
        delta = step - self.last_step
        if delta > 0:
            self.bar.update(delta)
            self.last_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        self._update_time_postfix(state)

        epoch = logs.get("epoch")
        if isinstance(epoch, (int, float)):
            self.postfix["epoch"] = f"{epoch:.2f}"

        if "loss" in logs:
            self.postfix["loss"] = f"{logs.get('loss', 0.0):.4f}"
            lr = logs.get("learning_rate")
            if isinstance(lr, (int, float)):
                self.postfix["lr"] = f"{lr:.2e}"
            grad = logs.get("grad_norm")
            if isinstance(grad, (int, float)):
                self.postfix["grad"] = f"{grad:.1f}"

        if "eval_loss" in logs:
            self.postfix["eval_loss"] = f"{logs.get('eval_loss', 0.0):.4f}"
            self.postfix["acc"] = f"{logs.get('eval_boundary_accuracy', 0.0):.4f}"
            self.postfix["punct_f1"] = f"{logs.get('eval_punct_macro_f1', 0.0):.4f}"
            self.postfix["punct_micro"] = f"{logs.get('eval_punct_micro_f1', 0.0):.4f}"

        if "train_runtime" in logs:
            self.postfix["train_loss"] = f"{logs.get('train_loss', 0.0):.4f}"
            self.postfix["steps_s"] = f"{logs.get('train_steps_per_second', 0.0):.2f}"

        if self.bar:
            self.bar.set_postfix(self.postfix, refresh=True)

    def _update_time_postfix(self, state) -> None:
        if not self.start_time:
            return
        elapsed = time.time() - self.start_time
        self.postfix["elapsed"] = format_seconds(elapsed)

        total_steps = int(state.max_steps) if int(state.max_steps) > 0 else 0
        current_step = max(int(state.global_step), 0)
        if total_steps > 0 and current_step > 0:
            seconds_per_step = elapsed / current_step
            remaining_steps = max(total_steps - current_step, 0)
            self.postfix["eta"] = format_seconds(seconds_per_step * remaining_steps)
        elif total_steps > 0:
            self.postfix["eta"] = "calculating"

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar:
            self.bar.close()
            self.bar = None


class LineTelemetryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        epoch = logs.get("epoch")
        epoch_text = f" epoch={epoch:.2f}" if isinstance(epoch, (int, float)) else ""

        if "eval_loss" in logs:
            print(
                "[eval] "
                f"step={step}{epoch_text} "
                f"loss={logs.get('eval_loss', 0.0):.4f} "
                f"acc={logs.get('eval_boundary_accuracy', 0.0):.4f} "
                f"punct_f1={logs.get('eval_punct_macro_f1', 0.0):.4f} "
                f"punct_micro={logs.get('eval_punct_micro_f1', 0.0):.4f} "
                f"runtime={logs.get('eval_runtime', 0.0):.1f}s"
            )
            return

        if "loss" in logs:
            lr = logs.get("learning_rate")
            lr_text = f" lr={lr:.2e}" if isinstance(lr, (int, float)) else ""
            grad = logs.get("grad_norm")
            grad_text = f" grad={grad:.2f}" if isinstance(grad, (int, float)) else ""
            print(
                "[train] "
                f"step={step}{epoch_text} "
                f"loss={logs.get('loss', 0.0):.4f}"
                f"{lr_text}{grad_text}"
            )
            return

        if "train_runtime" in logs:
            print(
                "[train-done] "
                f"runtime={logs.get('train_runtime', 0.0):.1f}s "
                f"samples/s={logs.get('train_samples_per_second', 0.0):.2f} "
                f"steps/s={logs.get('train_steps_per_second', 0.0):.2f} "
                f"loss={logs.get('train_loss', 0.0):.4f}"
            )


class WeightedTokenClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    ml_root = default_ml_root()
    parser = argparse.ArgumentParser(description="Train Arabic punctuation restoration model.")
    parser.add_argument("--train-file", default=os.path.join(ml_root, "data", "processed", "punctuation_train.jsonl"))
    parser.add_argument("--eval-file", default=os.path.join(ml_root, "data", "processed", "punctuation_eval.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(ml_root, "models", "punctuation_restorer"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--max-class-weight", type=float, default=12.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true", help="Use compact line logs instead of the progress bar.")
    parser.add_argument("--progress-bar", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def label_counts_from_file(path: str, max_examples: int = 0) -> Counter:
    counts: Counter = Counter()
    for idx, row in enumerate(read_jsonl(path), start=1):
        if max_examples and idx > max_examples:
            break
        counts.update(row["labels"])
    return counts


def class_weights_from_counts(counts: Counter, max_weight: float) -> torch.Tensor:
    total = sum(counts.values())
    weights = []
    for label in LABELS:
        count = max(int(counts.get(label, 0)), 1)
        weight = total / (len(LABELS) * count)
        weights.append(min(float(weight), max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    timer = Timer()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.train_file) or not os.path.exists(args.eval_file):
        raise FileNotFoundError(
            "Processed dataset is missing. Run ML_Tashkeel\\scripts\\build_punctuation_dataset.py first."
        )

    print("Arabic punctuation restoration training")
    print(f"Base model:       {args.model_name}")
    print(f"Train file:       {args.train_file}")
    print(f"Eval file:        {args.eval_file}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch sizes:      train={args.batch_size}, eval={args.eval_batch_size}")
    print(f"Precision:        bf16={args.bf16}, fp16={args.fp16}")
    print()

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name)

    train_dataset = PunctuationTokenDataset(
        args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_words=args.max_words,
        max_examples=args.max_train_examples or None,
    )
    eval_dataset = PunctuationTokenDataset(
        args.eval_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_words=args.max_words,
        max_examples=args.max_eval_examples or None,
    )

    label_counts = label_counts_from_file(args.train_file, args.max_train_examples)
    class_weights = None if args.no_class_weights else class_weights_from_counts(label_counts, args.max_class_weight)

    print("Dataset telemetry")
    print(f"  train rows/chunks: {train_dataset.row_count}/{len(train_dataset)}")
    print(f"  eval rows/chunks:  {eval_dataset.row_count}/{len(eval_dataset)}")
    print("  train label counts:")
    for label in LABELS:
        print(f"    {label:12s} {int(label_counts.get(label, 0))}")
    if class_weights is not None:
        print("  class weights:")
        for label, weight in zip(LABELS, class_weights.tolist()):
            print(f"    {label:12s} {weight:.4f}")
    print()

    save_label_files(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="punct_macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_workers,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        disable_tqdm=True,
    )

    trainer = WeightedTokenClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=trainer_compute_metrics,
        class_weights=class_weights,
    )
    trainer.remove_callback(PrinterCallback)
    if args.no_progress_bar:
        trainer.add_callback(LineTelemetryCallback)
    else:
        trainer.add_callback(ProgressTelemetryCallback)

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_label_files(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "final_train_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print()
    print("Training complete.")
    print(f"Elapsed:      {format_seconds(timer.elapsed())}")
    print(f"Model dir:    {args.output_dir}")
    print(f"Final metrics:{metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
