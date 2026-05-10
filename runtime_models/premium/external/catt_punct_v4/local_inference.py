from __future__ import annotations

import glob
import os
import re
import sys
import time
from dataclasses import dataclass

import torch

from checkpoint_compat import load_state_dict_with_vocab_expansion
from eo_runtime import TashkeelModel
from hybrid_splitter import HybridTextSplitter, SplitConfig, default_boundary_model_dir
from tashkeel_tokenizer import TashkeelTokenizer


CATT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CATT_ROOT, "..", ".."))
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_TYPE = "eo"
NEWLINE_RE = re.compile(r"(\n+)")
TOKEN_RE = re.compile(r"\S+\s*|\s+")
PUNCTUATION_CHUNK_RE = re.compile(r".+?(?:[،؛:,.!?؟]+(?:\s+|$)|$)", re.S)
ARABIC_BASE_RE = re.compile(r"[\u0621-\u064A\u0671]")
PAUSE_PUNCTUATION_RE = re.compile(r"([\u0621-\u064A\u0671])([\u064B-\u0652\u0670]+)(?=[.!?\u061F\u061B;:])")
PAUSE_STRIP_MARKS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0652")


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_BASE_RE.search(text))


def suppress_pause_punctuation_harakat(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        kept = "".join(mark for mark in match.group(2) if mark not in PAUSE_STRIP_MARKS)
        return match.group(1) + kept

    return PAUSE_PUNCTUATION_RE.sub(replace, text)


@dataclass
class SegmentTelemetry:
    original_segments: int
    inferred_segments: int
    chunked_segments: int
    warnings: list[str]
    split_methods: dict[str, int]
    max_segment_chars: int
    avg_segment_chars: float
    ml_boundary_used: int
    ml_boundary_candidates: int
    splitter: dict


class LocalCATTInference:
    def __init__(
        self,
        checkpoint: str | None = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        split_long_text: bool = True,
        split_target_chars: int = 320,
        split_hard_chars: int = 520,
        split_min_chars: int = 90,
        split_soft_target_words: int = 15,
        split_soft_min_words: int = 5,
        ml_boundary_split: bool = True,
        boundary_model_dir: str | None = None,
        boundary_threshold: float = 0.80,
        device: str | None = None,
    ):
        self.catt_dir = CATT_ROOT
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.checkpoint = os.path.abspath(checkpoint) if checkpoint else self._find_default_checkpoint()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.device_preference = (device or "auto").lower()
        boundary_model_dir = boundary_model_dir or default_boundary_model_dir(REPO_ROOT)
        effective_target_chars = split_target_chars if split_long_text else max_seq_len
        effective_hard_chars = split_hard_chars if split_long_text else max_seq_len
        self.split_config = SplitConfig(
            enabled=split_long_text,
            target_chars=effective_target_chars,
            hard_chars=min(effective_hard_chars, max_seq_len),
            min_chars=split_min_chars,
            soft_target_words=split_soft_target_words if split_long_text else 0,
            soft_min_words=split_soft_min_words,
            ml_enabled=ml_boundary_split and split_long_text,
            ml_threshold=boundary_threshold,
            boundary_model_dir=boundary_model_dir,
        )
        self.splitter: HybridTextSplitter | None = None

    def _find_default_checkpoint(self) -> str:
        preferred = [
            os.path.join(self.catt_dir, "catt_eo_ft_balanced_v4_punct", "catt_eo_ft_best.pt"),
            os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v3", "catt_eo_ft_best.pt"),
        ]
        for path in preferred:
            if os.path.exists(path):
                return os.path.abspath(path)

        for folder in [
            os.path.join(self.catt_dir, "catt_eo_ft_balanced_v4_punct"),
            os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v3"),
        ]:
            matches = sorted(glob.glob(os.path.join(folder, "catt_eo_ft-epoch=*.ckpt")), key=os.path.getmtime, reverse=True)
            if matches:
                return os.path.abspath(matches[0])

        raise FileNotFoundError("Could not auto-detect a v4 punctuation-aware checkpoint.")

    def _choose_device(self):
        if self.device_preference in {"cpu"}:
            return torch.device("cpu")
        if self.device_preference in {"gpu", "cuda"}:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU was requested, but CUDA is not available in this runtime.")
            return torch.device("cuda")
        if self.device_preference not in {"auto", ""}:
            raise RuntimeError(f"Unknown device preference: {self.device_preference}")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self):
        if self.model is not None:
            return

        if self.catt_dir not in sys.path:
            sys.path.insert(0, self.catt_dir)

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        tokenizer = TashkeelTokenizer()
        model = TashkeelModel(tokenizer, max_seq_len=self.max_seq_len, n_layers=6, learnable_pos_emb=False)
        device = self._choose_device()
        self.device = device.type

        state = torch.load(self.checkpoint, map_location="cpu")
        missing, unexpected = load_state_dict_with_vocab_expansion(model, state)
        if unexpected:
            print(f"[v4] warning: unexpected checkpoint keys={len(unexpected)}")
        if missing:
            print(f"[v4] warning: missing checkpoint keys={len(missing)}")

        model.eval().to(device)
        self.tokenizer = tokenizer
        self.model = model
        self.splitter = HybridTextSplitter(self.cleaned_length, self.split_config)

    def cleaned_length(self, text: str) -> int:
        return len(self.tokenizer.clean_text(text))

    def _split_long_segment(self, text: str) -> list[str]:
        if self.cleaned_length(text) <= self.max_seq_len:
            return [text]

        chunks = self._split_by_punctuation(text)
        if chunks:
            return chunks
        return self._split_by_tokens(text)

    def _split_by_punctuation(self, text: str) -> list[str]:
        units = [match.group(0) for match in PUNCTUATION_CHUNK_RE.finditer(text) if match.group(0).strip()]
        if len(units) <= 1:
            return []
        return self._pack_units(units)

    def _pack_units(self, units: list[str]) -> list[str]:
        chunks = []
        current = ""
        for unit in units:
            if self.cleaned_length(unit) > self.max_seq_len:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_by_tokens(unit))
                continue

            candidate = current + unit
            if current and self.cleaned_length(candidate) > self.max_seq_len:
                chunks.append(current)
                current = unit
            else:
                current = candidate
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk.strip()]

    def _split_by_tokens(self, text: str) -> list[str]:
        chunks = []
        current = ""
        for token in TOKEN_RE.findall(text):
            candidate = current + token
            if current and self.cleaned_length(candidate) > self.max_seq_len:
                chunks.append(current)
                current = token
                if self.cleaned_length(current) > self.max_seq_len:
                    chunks.extend(self._split_hard(current))
                    current = ""
            else:
                current = candidate
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk.strip()]

    def _split_hard(self, text: str) -> list[str]:
        pieces = []
        current = ""
        for ch in text:
            candidate = current + ch
            if current and self.cleaned_length(candidate) > self.max_seq_len:
                pieces.append(current)
                current = ch
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _prepare_segments(self, text: str):
        original_segments = 0
        chunked_segments = 0
        segments = []
        warnings: list[str] = []
        split_methods: dict[str, int] = {}
        max_segment_chars = 0
        total_segment_chars = 0
        measured_segments = 0
        ml_boundary_used = 0
        ml_boundary_candidates = 0

        if self.splitter is None:
            self.splitter = HybridTextSplitter(self.cleaned_length, self.split_config)

        for part in NEWLINE_RE.split(text):
            if not part:
                continue
            if NEWLINE_RE.fullmatch(part):
                segments.append(("raw", part))
                continue
            original_segments += 1
            if not contains_arabic(part):
                segments.append(("raw", part))
                continue

            leading_len = len(part) - len(part.lstrip())
            trailing_len = len(part) - len(part.rstrip())
            leading = part[:leading_len]
            trailing = part[len(part) - trailing_len :] if trailing_len else ""
            body = part.strip()
            if leading:
                segments.append(("raw", leading))

            pieces, stats = self.splitter.split(body)
            if len(pieces) > 1:
                chunked_segments += 1
            warnings.extend(stats.warnings)
            ml_boundary_used += stats.ml_boundary_used
            ml_boundary_candidates += stats.ml_boundary_candidates

            for method, count in stats.method_counts.items():
                split_methods[method] = split_methods.get(method, 0) + int(count)
            max_segment_chars = max(max_segment_chars, stats.max_cleaned_length)
            if pieces:
                total_segment_chars += int(round(stats.avg_cleaned_length * len(pieces)))
                measured_segments += len(pieces)

            for index, piece in enumerate(pieces):
                if index > 0:
                    segments.append(("raw", self.split_config.joiner))
                segments.append(("infer", piece.text))

            if trailing:
                segments.append(("raw", trailing))

        avg_segment_chars = total_segment_chars / measured_segments if measured_segments else 0.0
        splitter_status = self.splitter.status() if self.splitter else {}
        return segments, SegmentTelemetry(
            original_segments=original_segments,
            inferred_segments=sum(1 for kind, _ in segments if kind == "infer"),
            chunked_segments=chunked_segments,
            warnings=warnings,
            split_methods=split_methods,
            max_segment_chars=max_segment_chars,
            avg_segment_chars=avg_segment_chars,
            ml_boundary_used=ml_boundary_used,
            ml_boundary_candidates=ml_boundary_candidates,
            splitter=splitter_status,
        )

    def diacritize(self, text: str):
        self.load()
        t0 = time.time()
        segments, segment_info = self._prepare_segments(text)
        infer_segments = [segment for kind, segment in segments if kind == "infer"]

        predictions = []
        for start in range(0, len(infer_segments), self.batch_size):
            batch = infer_segments[start : start + self.batch_size]
            outputs = self.model.do_tashkeel_batch(batch, batch_size=len(batch), verbose=False)
            predictions.extend(outputs)

        output_parts = []
        pred_index = 0
        for kind, segment in segments:
            if kind == "raw":
                output_parts.append(segment)
                continue
            output_parts.append(predictions[pred_index])
            pred_index += 1

        return suppress_pause_punctuation_harakat("".join(output_parts)), {
            "elapsed": time.time() - t0,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "segment_info": segment_info,
        }
