from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


LABELS = [
    "NONE",
    "COMMA",
    "PERIOD",
    "QUESTION",
    "COLON",
    "SEMICOLON",
    "EXCLAMATION",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

LABEL_TO_OUTPUT = {
    "NONE": "",
    "COMMA": "\u060c",
    "PERIOD": ".",
    "QUESTION": "\u061f",
    "COLON": ":",
    "SEMICOLON": "\u061b",
    "EXCLAMATION": "!",
}

PUNCT_CHAR_TO_LABEL = {
    "\u060c": "COMMA",
    ",": "COMMA",
    ".": "PERIOD",
    "\u06d4": "PERIOD",
    "?": "QUESTION",
    "\u061f": "QUESTION",
    ":": "COLON",
    ";": "SEMICOLON",
    "\u061b": "SEMICOLON",
    "!": "EXCLAMATION",
}

TARGET_PUNCT_CHARS = set(PUNCT_CHAR_TO_LABEL)

DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
TATWEEL = "\u0640"
ARABIC_WORD_RE = re.compile(r"[\u0621-\u064a\u0671]+")
PROMPT_PREFIX_RE = re.compile(r"^\s*\S+\s+\S+:\s*", re.S)


@dataclass(frozen=True)
class BoundaryExample:
    source: str
    words: list[str]
    labels: list[str]
    text_unpunctuated: str
    text_punctuated: str


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text).replace(TATWEEL, "")


def punctuation_label_from_gap(text: str) -> str:
    """Return the rightmost supported punctuation label in a word gap."""

    for char in reversed(text):
        label = PUNCT_CHAR_TO_LABEL.get(char)
        if label:
            return label
    return "NONE"


def extract_words_and_labels(text: str, source: str = "") -> BoundaryExample | None:
    plain = strip_diacritics(text)
    matches = list(ARABIC_WORD_RE.finditer(plain))
    if not matches:
        return None

    words: list[str] = []
    labels: list[str] = []

    for idx, match in enumerate(matches):
        word = match.group(0)
        words.append(word)

        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(plain)
        gap = plain[match.end() : next_start]
        labels.append(punctuation_label_from_gap(gap))

    if len(words) != len(labels):
        raise ValueError("Internal word/label alignment mismatch.")

    if not words:
        return None

    return BoundaryExample(
        source=source,
        words=words,
        labels=labels,
        text_unpunctuated=" ".join(words),
        text_punctuated=reconstruct_text(words, labels),
    )


def reconstruct_text(words: list[str], labels: list[str]) -> str:
    if len(words) != len(labels):
        raise ValueError("words and labels must have the same length.")
    pieces = []
    for word, label in zip(words, labels):
        pieces.append(f"{word}{LABEL_TO_OUTPUT.get(label, '')}")
    return " ".join(pieces)


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def copy_if_needed(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    shutil.copy2(src, dst)


def count_labels(rows: Iterable[dict]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        counter.update(row["labels"])
    return counter


def iter_word_chunks(
    words: list[str],
    labels: list[str],
    max_words: int,
) -> Iterable[tuple[list[str], list[str]]]:
    if len(words) != len(labels):
        raise ValueError("words and labels must have the same length.")
    if max_words <= 0:
        yield words, labels
        return
    for start in range(0, len(words), max_words):
        yield words[start : start + max_words], labels[start : start + max_words]


def default_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_ml_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

