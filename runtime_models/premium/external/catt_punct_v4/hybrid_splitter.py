from __future__ import annotations

import os
import re
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Skipping import of cpp extensions.*")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")


ARABIC_WORD_RE = re.compile(r"[\u0621-\u064a\u0671]+")
DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
STRONG_BOUNDARY_RE = re.compile(r".+?(?:[\.\!\?\u061f\u061b;:]+(?:\s+|$)|$)", re.S)
SOFT_BOUNDARY_RE = re.compile(r".+?(?:[\u060c,]+(?:\s+|$)|$)", re.S)
TOKEN_RE = re.compile(r"\S+")
PUNCT_RE = re.compile(r"[\u060c,\.\!\?\u061f\u061b;:]")


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text).replace("\u0640", "")


def default_boundary_model_dir(repo_root: str) -> str:
    return os.path.abspath(
        os.path.join(
            repo_root,
            "..",
            "ML_Tashkeel",
            "models",
            "punctuation_boundary_detector_v4_v6_final_from_v3_bs112_lr2e6_e1",
        )
    )


@dataclass
class SplitConfig:
    enabled: bool = True
    target_chars: int = 320
    hard_chars: int = 520
    min_chars: int = 90
    soft_target_words: int = 15
    soft_min_words: int = 5
    joiner: str = " "
    ml_enabled: bool = True
    ml_threshold: float = 0.80
    ml_max_length: int = 256
    ml_max_words: int = 180
    ml_batch_size: int = 16
    boundary_model_dir: str | None = None


@dataclass
class SplitPiece:
    text: str
    method: str


@dataclass
class SplitStats:
    original_cleaned_length: int = 0
    max_cleaned_length: int = 0
    avg_cleaned_length: float = 0.0
    method_counts: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    ml_boundary_used: int = 0
    ml_boundary_candidates: int = 0


class BoundaryPredictor:
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.80,
        max_length: int = 256,
        max_words: int = 180,
        batch_size: int = 16,
    ) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self.threshold = threshold
        self.max_length = max_length
        self.max_words = max_words
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return os.path.exists(os.path.join(self.model_dir, "config.json")) and os.path.exists(
            os.path.join(self.model_dir, "model.safetensors")
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> bool:
        if self.loaded:
            return True
        if not self.available:
            self.load_error = f"Boundary model not found: {self.model_dir}"
            return False

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
            if not getattr(tokenizer, "is_fast", False):
                self.load_error = "Boundary model tokenizer is not fast; word alignment is unavailable."
                return False

            model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            model.eval().to(device)
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            self.load_error = None
            return True
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _iter_word_chunks(self, words: list[str]):
        for start in range(0, len(words), self.max_words):
            yield start, words[start : start + self.max_words]

    def punct_probabilities(self, words: list[str]) -> list[float]:
        if not words:
            return []
        if not self.load():
            return []

        import torch

        probabilities = [0.0] * len(words)
        with torch.inference_mode():
            for chunk_start, chunk_words in self._iter_word_chunks(words):
                encoded = self.tokenizer(
                    chunk_words,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                word_ids = encoded.word_ids(batch_index=0)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits[0], dim=-1)

                for token_idx, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue
                    next_word_id = word_ids[token_idx + 1] if token_idx + 1 < len(word_ids) else None
                    if next_word_id == word_id:
                        continue
                    absolute_word_id = chunk_start + word_id
                    if absolute_word_id < len(probabilities):
                        probabilities[absolute_word_id] = float(probs[token_idx, 1].item())

        return probabilities


class HybridTextSplitter:
    def __init__(self, cleaned_length: Callable[[str], int], config: SplitConfig) -> None:
        self.cleaned_length = cleaned_length
        self.config = config
        self.boundary_predictor: BoundaryPredictor | None = None
        if config.ml_enabled and config.boundary_model_dir:
            self.boundary_predictor = BoundaryPredictor(
                model_dir=config.boundary_model_dir,
                threshold=config.ml_threshold,
                max_length=config.ml_max_length,
                max_words=config.ml_max_words,
                batch_size=config.ml_batch_size,
            )

    def status(self) -> dict:
        predictor = self.boundary_predictor
        return {
            "enabled": self.config.enabled,
            "strategy": "strong_punct_then_word_refine" if self.config.enabled else "disabled_maxseq_only",
            "target_chars": self.config.target_chars,
            "hard_chars": self.config.hard_chars,
            "soft_target_words": self.config.soft_target_words,
            "soft_min_words": self.config.soft_min_words,
            "ml_enabled": self.config.ml_enabled,
            "ml_threshold": self.config.ml_threshold,
            "boundary_model_dir": predictor.model_dir if predictor else None,
            "boundary_model_available": predictor.available if predictor else False,
            "boundary_model_loaded": predictor.loaded if predictor else False,
            "boundary_model_error": predictor.load_error if predictor else None,
        }

    def split(self, text: str) -> tuple[list[SplitPiece], SplitStats]:
        body = text.strip()
        stats = SplitStats(original_cleaned_length=self.cleaned_length(body))
        if not body:
            return [], stats

        if not self.config.enabled and stats.original_cleaned_length <= self.config.target_chars:
            pieces = [SplitPiece(body, "none")]
            self._finish_stats(pieces, stats)
            return pieces, stats

        if stats.original_cleaned_length <= self.config.target_chars:
            pieces = [SplitPiece(body, "none")]
            self._finish_stats(pieces, stats)
            return pieces, stats

        if not self.config.enabled:
            pieces = self._split_with_tokens([SplitPiece(body, "none")])
            pieces = self._split_hard(pieces)
            self._finish_stats(pieces, stats)
            return pieces, stats

        pieces = self._split_with_punctuation([SplitPiece(body, "none")])
        pieces = self._split_with_ml_boundaries(pieces, stats)
        pieces = self._split_with_tokens(pieces)
        pieces = self._split_hard(pieces)
        pieces = [piece for piece in pieces if piece.text.strip()]
        self._finish_stats(pieces, stats)
        return pieces, stats

    def _finish_stats(self, pieces: list[SplitPiece], stats: SplitStats) -> None:
        lengths = [self.cleaned_length(piece.text) for piece in pieces]
        stats.max_cleaned_length = max(lengths, default=0)
        stats.avg_cleaned_length = sum(lengths) / len(lengths) if lengths else 0.0
        stats.method_counts = Counter(piece.method for piece in pieces)

    def _needs_split(self, text: str) -> bool:
        return self.cleaned_length(text) > self.config.target_chars

    def _word_count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))

    def _needs_soft_refine(self, text: str) -> bool:
        return self._needs_split(text) or (
            self.config.soft_target_words > 0
            and self._word_count(text) > self.config.soft_target_words
        )

    def _delimiter_units(self, text: str, pattern: re.Pattern[str]) -> list[str]:
        units = [match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()]
        return units if len(units) > 1 else []

    def _split_with_delimiter(
        self,
        pieces: list[SplitPiece],
        pattern: re.Pattern[str],
        method: str,
    ) -> list[SplitPiece]:
        output: list[SplitPiece] = []
        for piece in pieces:
            if not self._needs_split(piece.text):
                output.append(piece)
                continue
            units = self._delimiter_units(piece.text, pattern)
            if not units:
                output.append(piece)
                continue
            output.extend(self._pack_units(units, method))
        return output

    def _split_with_punctuation(self, pieces: list[SplitPiece]) -> list[SplitPiece]:
        output: list[SplitPiece] = []
        for piece in pieces:
            units = self._delimiter_units(piece.text, STRONG_BOUNDARY_RE)
            if not units:
                output.append(piece)
                continue
            output.extend(SplitPiece(unit, "strong") for unit in units)
        return self._split_long_with_soft_punctuation(output)

    def _split_long_with_soft_punctuation(self, pieces: list[SplitPiece]) -> list[SplitPiece]:
        output: list[SplitPiece] = []
        for piece in pieces:
            if not self._needs_soft_refine(piece.text):
                output.append(piece)
                continue
            units = self._delimiter_units(piece.text, SOFT_BOUNDARY_RE)
            if not units:
                output.append(piece)
                continue
            output.extend(self._pack_soft_units(units))
        return output

    def _pack_soft_units(self, units: list[str]) -> list[SplitPiece]:
        packed: list[SplitPiece] = []
        current: list[str] = []
        for unit in units:
            if not current:
                current = [unit]
                continue

            candidate = " ".join(current + [unit])
            current_text = " ".join(current)
            candidate_too_long = (
                self.cleaned_length(candidate) > self.config.target_chars
                or (
                    self.config.soft_target_words > 0
                    and self._word_count(candidate) > self.config.soft_target_words
                )
            )
            current_big_enough = (
                self.cleaned_length(current_text) >= self.config.min_chars
                or self._word_count(current_text) >= self.config.soft_min_words
            )
            if candidate_too_long and current_big_enough:
                packed.append(SplitPiece(current_text, "soft"))
                current = [unit]
            else:
                current.append(unit)

        if current:
            packed.append(SplitPiece(" ".join(current), "soft"))
        return packed

    def _pack_units(self, units: list[str], method: str) -> list[SplitPiece]:
        packed: list[SplitPiece] = []
        current: list[str] = []
        for unit in units:
            if not current:
                current = [unit]
                continue

            candidate = " ".join(current + [unit])
            current_text = " ".join(current)
            if (
                self.cleaned_length(candidate) > self.config.target_chars
                and self.cleaned_length(current_text) >= self.config.min_chars
            ):
                packed.append(SplitPiece(current_text, method))
                current = [unit]
            else:
                current.append(unit)

        if current:
            packed.append(SplitPiece(" ".join(current), method))
        return packed

    def _split_with_ml_boundaries(self, pieces: list[SplitPiece], stats: SplitStats) -> list[SplitPiece]:
        if not self.config.ml_enabled:
            return pieces

        output: list[SplitPiece] = []
        warned = False
        for piece in pieces:
            if not self._needs_split(piece.text):
                output.append(piece)
                continue
            if len(PUNCT_RE.findall(piece.text)) >= 2:
                output.append(piece)
                continue
            if not self.boundary_predictor:
                output.append(piece)
                continue
            ml_pieces = self._ml_split_piece(piece.text, stats)
            if len(ml_pieces) <= 1:
                if self.boundary_predictor.load_error and not warned:
                    stats.warnings.append(f"ML boundary splitter unavailable: {self.boundary_predictor.load_error}")
                    warned = True
                output.append(piece)
            else:
                output.extend(ml_pieces)
        return output

    def _ml_split_piece(self, text: str, stats: SplitStats) -> list[SplitPiece]:
        words = ARABIC_WORD_RE.findall(strip_diacritics(text))
        if len(words) < 8 or not self.boundary_predictor:
            return [SplitPiece(text, "none")]

        probabilities = self.boundary_predictor.punct_probabilities(words)
        if not probabilities:
            return [SplitPiece(text, "none")]

        stats.ml_boundary_candidates += sum(1 for prob in probabilities if prob >= self.config.ml_threshold)
        chunks: list[list[str]] = []
        current: list[str] = []
        best_candidate: int | None = None
        best_candidate_distance = 10**9

        for idx, word in enumerate(words):
            current.append(word)
            current_text = " ".join(current)
            current_len = self.cleaned_length(current_text)
            prob = probabilities[idx]

            if idx < len(words) - 1 and prob >= self.config.ml_threshold and current_len >= self.config.min_chars:
                distance = abs(self.config.target_chars - current_len)
                if distance <= best_candidate_distance:
                    best_candidate = len(current)
                    best_candidate_distance = distance

            if current_len >= self.config.target_chars:
                split_at = best_candidate
                if split_at is None and current_len >= self.config.hard_chars:
                    split_at = len(current)
                if split_at is not None and split_at > 0:
                    chunks.append(current[:split_at])
                    current = current[split_at:]
                    best_candidate = None
                    best_candidate_distance = 10**9

        if current:
            chunks.append(current)

        if len(chunks) <= 1:
            return [SplitPiece(text, "none")]

        stats.ml_boundary_used += len(chunks) - 1
        return [SplitPiece(" ".join(chunk), "ml") for chunk in chunks if chunk]

    def _split_with_tokens(self, pieces: list[SplitPiece]) -> list[SplitPiece]:
        output: list[SplitPiece] = []
        for piece in pieces:
            if not self._needs_split(piece.text):
                output.append(piece)
                continue
            tokens = TOKEN_RE.findall(piece.text)
            if len(tokens) <= 1:
                output.append(piece)
                continue
            output.extend(self._pack_units(tokens, "token"))
        return output

    def _split_hard(self, pieces: list[SplitPiece]) -> list[SplitPiece]:
        output: list[SplitPiece] = []
        for piece in pieces:
            if self.cleaned_length(piece.text) <= self.config.hard_chars:
                output.append(piece)
                continue
            current = ""
            for char in piece.text:
                candidate = current + char
                if current and self.cleaned_length(candidate) > self.config.hard_chars:
                    output.append(SplitPiece(current.strip(), "hard"))
                    current = char
                else:
                    current = candidate
            if current.strip():
                output.append(SplitPiece(current.strip(), "hard"))
        return output
