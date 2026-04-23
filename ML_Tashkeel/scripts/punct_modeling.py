from __future__ import annotations

import logging
import json
import os
import time
import warnings
from collections import Counter
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from punct_common import ID_TO_LABEL, LABELS, LABEL_TO_ID, iter_word_chunks

hf_logging.set_verbosity_error()


class PunctuationTokenDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 256,
        max_words: int = 180,
        max_examples: int | None = None,
    ) -> None:
        self.features: list[dict[str, Any]] = []
        self.label_counts: Counter = Counter()
        self.row_count = 0
        self.chunk_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if max_examples is not None and self.row_count >= max_examples:
                    break
                row = json.loads(line)
                self.row_count += 1
                words = row["words"]
                labels = row["labels"]
                self.label_counts.update(labels)

                for chunk_words, chunk_labels in iter_word_chunks(words, labels, max_words):
                    encoded = encode_words_and_labels(
                        tokenizer=tokenizer,
                        words=chunk_words,
                        labels=chunk_labels,
                        max_length=max_length,
                    )
                    if any(label != -100 for label in encoded["labels"]):
                        self.features.append(encoded)
                        self.chunk_count += 1

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def encode_words_and_labels(tokenizer, words: list[str], labels: list[str], max_length: int) -> dict:
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    word_ids = encoded.word_ids()
    aligned_labels: list[int] = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            aligned_labels.append(-100)
            continue
        next_word_id = word_ids[idx + 1] if idx + 1 < len(word_ids) else None
        if next_word_id != word_id:
            aligned_labels.append(LABEL_TO_ID[labels[word_id]])
        else:
            aligned_labels.append(-100)
    encoded["labels"] = aligned_labels
    return dict(encoded)


def compute_metrics_from_arrays(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    mask = labels != -100
    y_true = labels[mask]
    y_pred = predictions[mask]
    if y_true.size == 0:
        return {
            "boundary_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "punct_macro_f1": 0.0,
            "punct_micro_f1": 0.0,
        }

    punct_ids = [LABEL_TO_ID[label] for label in LABELS if label != "NONE"]
    return {
        "boundary_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "punct_macro_f1": float(
            f1_score(y_true, y_pred, labels=punct_ids, average="macro", zero_division=0)
        ),
        "punct_micro_f1": float(
            f1_score(y_true, y_pred, labels=punct_ids, average="micro", zero_division=0)
        ),
    }


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_metrics_from_arrays(labels, predictions)


def load_tokenizer(model_name_or_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"{model_name_or_dir} did not load a fast tokenizer; word-boundary alignment needs one."
        )
    return tokenizer


def load_model(model_name_or_dir: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_dir,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict_labels_for_words(
    model,
    tokenizer,
    words: list[str],
    max_length: int = 256,
    max_words: int = 180,
    batch_size: int = 32,
    none_threshold: float = 0.0,
    device: torch.device | None = None,
) -> tuple[list[str], list[float]]:
    if not words:
        return [], []

    device = device or choose_device()
    model.to(device)
    model.eval()

    chunk_payloads: list[tuple[list[str], dict]] = []
    for chunk_words, _ in iter_word_chunks(words, ["NONE"] * len(words), max_words):
        encoded = tokenizer(
            chunk_words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        chunk_payloads.append((chunk_words, encoded))

    predicted_labels: list[str] = []
    confidences: list[float] = []

    for start in range(0, len(chunk_payloads), batch_size):
        batch_payloads = chunk_payloads[start : start + batch_size]
        # Different chunks may have different lengths; run them one-by-one for
        # predictable word_id handling during inference.
        for chunk_words, encoded in batch_payloads:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits[0], dim=-1)
            word_ids = tokenizer(
                chunk_words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).word_ids()

            chunk_labels = ["NONE"] * len(chunk_words)
            chunk_conf = [0.0] * len(chunk_words)
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                next_word_id = word_ids[token_idx + 1] if token_idx + 1 < len(word_ids) else None
                if next_word_id == word_id:
                    continue
                token_probs = probs[token_idx]
                pred_id = int(torch.argmax(token_probs).item())
                confidence = float(token_probs[pred_id].item())
                label = ID_TO_LABEL[pred_id]
                if label != "NONE" and confidence < none_threshold:
                    label = "NONE"
                chunk_labels[word_id] = label
                chunk_conf[word_id] = confidence

            predicted_labels.extend(chunk_labels)
            confidences.extend(chunk_conf)

    return predicted_labels[: len(words)], confidences[: len(words)]


def save_label_files(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"labels": LABELS, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL},
            handle,
            ensure_ascii=False,
            indent=2,
        )


def format_seconds(seconds: float) -> str:
    total_seconds = int(round(seconds))
    if total_seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {rem}s"


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start
