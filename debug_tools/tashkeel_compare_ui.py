from __future__ import annotations

import difflib
import sys
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


TASHKEEL_CODEPOINTS = set(range(0x064B, 0x0653)) | {0x0670}
ARABIC_LETTER_START = 0x0621
ARABIC_LETTER_END = 0x064A
ALIF_WASLA = "\u0671"
ALIF = "\u0627"
TATWEEL = "\u0640"
MAX_DP_CELLS = 8_000_000


@dataclass
class WordToken:
    text: str
    base: str
    start_pair: int
    end_pair: int


@dataclass
class ParsedText:
    words: list[WordToken]
    pairs: list[tuple[str, str]]
    pair_to_word: list[int]
    final_pairs: set[int]


@dataclass
class WordDifference:
    correct_word: str
    model_word: str
    differences: int
    first_pair_index: int


@dataclass
class ComparisonResult:
    tashkeel_differences: int
    marked_positions: int
    matched_letters: int
    correct_letters: int
    model_letters: int
    base_substitutions: int
    unmatched_correct_letters: int
    unmatched_model_letters: int
    no_case_differences: int
    no_case_marked_positions: int
    word_differences: list[WordDifference]


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def is_tashkeel(ch: str) -> bool:
    return ord(ch) in TASHKEEL_CODEPOINTS


def is_arabic_letter(ch: str) -> bool:
    return ARABIC_LETTER_START <= ord(ch) <= ARABIC_LETTER_END or ch == ALIF_WASLA


def normalize_base(ch: str) -> str:
    return ALIF if ch == ALIF_WASLA else ch


def normalize_diacritics(diacritics: str) -> str:
    return "".join(sorted(diacritics))


def split_word_pairs(word: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    base: str | None = None
    diacritics = ""

    for ch in word.replace(TATWEEL, ""):
        if is_tashkeel(ch):
            if base is not None:
                diacritics += ch
        elif is_arabic_letter(ch):
            if base is not None:
                pairs.append((normalize_base(base), diacritics))
            base = ch
            diacritics = ""

    if base is not None:
        pairs.append((normalize_base(base), diacritics))

    return pairs


def parse_arabic_text(text: str) -> ParsedText:
    words: list[WordToken] = []
    pairs: list[tuple[str, str]] = []
    pair_to_word: list[int] = []
    final_pairs: set[int] = set()
    current_chars: list[str] = []

    def flush_word() -> None:
        nonlocal current_chars
        if not current_chars:
            return

        raw_word = "".join(current_chars)
        word_pairs = split_word_pairs(raw_word)
        current_chars = []
        if not word_pairs:
            return

        word_index = len(words)
        start_pair = len(pairs)
        base = "".join(base_char for base_char, _ in word_pairs)
        pairs.extend(word_pairs)
        pair_to_word.extend([word_index] * len(word_pairs))
        end_pair = len(pairs)
        final_pairs.add(end_pair - 1)
        words.append(WordToken(raw_word, base, start_pair, end_pair))

    for ch in text:
        if is_arabic_letter(ch):
            current_chars.append(ch)
        elif is_tashkeel(ch):
            if current_chars:
                current_chars.append(ch)
        elif ch == TATWEEL:
            if current_chars:
                current_chars.append(ch)
        else:
            flush_word()

    flush_word()
    return ParsedText(words, pairs, pair_to_word, final_pairs)


def align_with_dp(correct_bases: str, model_bases: str) -> list[tuple[str, int | None, int | None]]:
    n = len(correct_bases)
    m = len(model_bases)
    previous = list(range(m + 1))
    backtrace: list[bytearray] = []

    # 0 = match, 1 = substitution, 2 = delete, 3 = insert.
    for i in range(1, n + 1):
        current = [i] + [0] * m
        row = bytearray(m + 1)
        row[0] = 2

        for j in range(1, m + 1):
            if correct_bases[i - 1] == model_bases[j - 1]:
                diagonal_cost = previous[j - 1]
                diagonal_op = 0
            else:
                # Prefer insert+delete over aligning two different base letters.
                diagonal_cost = previous[j - 1] + 3
                diagonal_op = 1

            delete_cost = previous[j] + 1
            insert_cost = current[j - 1] + 1
            best_cost = min(diagonal_cost, delete_cost, insert_cost)
            current[j] = best_cost

            if best_cost == diagonal_cost:
                row[j] = diagonal_op
            elif best_cost == delete_cost:
                row[j] = 2
            else:
                row[j] = 3

        previous = current
        backtrace.append(row)

    operations: list[tuple[str, int | None, int | None]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        if i == 0:
            op = 3
        elif j == 0:
            op = 2
        else:
            op = backtrace[i - 1][j]

        if op == 0:
            operations.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == 1:
            operations.append(("substitute", i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == 2:
            operations.append(("delete", i - 1, None))
            i -= 1
        else:
            operations.append(("insert", None, j - 1))
            j -= 1

    operations.reverse()
    return operations


def align_with_sequence_matcher(
    correct_bases: str, model_bases: str
) -> list[tuple[str, int | None, int | None]]:
    operations: list[tuple[str, int | None, int | None]] = []
    matcher = difflib.SequenceMatcher(a=correct_bases, b=model_bases, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                operations.append(("match", i1 + offset, j1 + offset))
        elif tag == "delete":
            for i in range(i1, i2):
                operations.append(("delete", i, None))
        elif tag == "insert":
            for j in range(j1, j2):
                operations.append(("insert", None, j))
        else:
            for i in range(i1, i2):
                operations.append(("delete", i, None))
            for j in range(j1, j2):
                operations.append(("insert", None, j))

    return operations


def align_bases(correct_bases: str, model_bases: str) -> list[tuple[str, int | None, int | None]]:
    if len(correct_bases) * len(model_bases) <= MAX_DP_CELLS:
        return align_with_dp(correct_bases, model_bases)
    return align_with_sequence_matcher(correct_bases, model_bases)


def compare_tashkeel(correct_text: str, model_text: str) -> ComparisonResult:
    correct = parse_arabic_text(correct_text)
    model = parse_arabic_text(model_text)
    correct_bases = "".join(base for base, _ in correct.pairs)
    model_bases = "".join(base for base, _ in model.pairs)
    operations = align_bases(correct_bases, model_bases)

    tashkeel_differences = 0
    marked_positions = 0
    matched_letters = 0
    base_substitutions = 0
    unmatched_correct_letters = 0
    unmatched_model_letters = 0
    no_case_differences = 0
    no_case_marked_positions = 0
    word_difference_counts: dict[tuple[int, int], int] = defaultdict(int)
    word_difference_first_pair: dict[tuple[int, int], int] = {}

    for op, correct_index, model_index in operations:
        if op == "match":
            assert correct_index is not None and model_index is not None
            matched_letters += 1
            _, correct_diac = correct.pairs[correct_index]
            _, model_diac = model.pairs[model_index]
            correct_diac = normalize_diacritics(correct_diac)
            model_diac = normalize_diacritics(model_diac)

            has_mark = bool(correct_diac or model_diac)
            if has_mark:
                marked_positions += 1

            is_case_ending = (
                correct_index in correct.final_pairs or model_index in model.final_pairs
            )
            if not is_case_ending and has_mark:
                no_case_marked_positions += 1

            if correct_diac != model_diac:
                tashkeel_differences += 1
                if not is_case_ending:
                    no_case_differences += 1

                correct_word_index = correct.pair_to_word[correct_index]
                model_word_index = model.pair_to_word[model_index]
                key = (correct_word_index, model_word_index)
                word_difference_counts[key] += 1
                word_difference_first_pair.setdefault(key, correct_index)
        elif op == "substitute":
            base_substitutions += 1
        elif op == "delete":
            unmatched_correct_letters += 1
        elif op == "insert":
            unmatched_model_letters += 1

    word_differences = [
        WordDifference(
            correct_word=correct.words[correct_word_index].text,
            model_word=model.words[model_word_index].text,
            differences=count,
            first_pair_index=word_difference_first_pair[(correct_word_index, model_word_index)],
        )
        for (correct_word_index, model_word_index), count in word_difference_counts.items()
    ]
    word_differences.sort(key=lambda item: item.first_pair_index)

    return ComparisonResult(
        tashkeel_differences=tashkeel_differences,
        marked_positions=marked_positions,
        matched_letters=matched_letters,
        correct_letters=len(correct.pairs),
        model_letters=len(model.pairs),
        base_substitutions=base_substitutions,
        unmatched_correct_letters=unmatched_correct_letters,
        unmatched_model_letters=unmatched_model_letters,
        no_case_differences=no_case_differences,
        no_case_marked_positions=no_case_marked_positions,
        word_differences=word_differences,
    )


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.2f}%"


class TashkeelCompareUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Tashkeel Error Comparator")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)

        self._build_styles()
        self._build_layout()

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Summary.TLabel", font=("Segoe UI", 11))

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

        ttk.Label(main, text="Compare Arabic Tashkeel", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        model_frame = ttk.LabelFrame(main, text="Model output", padding=8)
        model_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        model_frame.rowconfigure(0, weight=1)
        model_frame.columnconfigure(0, weight=1)
        self.model_text = ScrolledText(model_frame, wrap="word", font=("Tahoma", 13), undo=True)
        self.model_text.grid(row=0, column=0, sticky="nsew")
        self.model_text.tag_configure("rtl", justify="right")

        correct_frame = ttk.LabelFrame(main, text="Correct output", padding=8)
        correct_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        correct_frame.rowconfigure(0, weight=1)
        correct_frame.columnconfigure(0, weight=1)
        self.correct_text = ScrolledText(correct_frame, wrap="word", font=("Tahoma", 13), undo=True)
        self.correct_text.grid(row=0, column=0, sticky="nsew")
        self.correct_text.tag_configure("rtl", justify="right")

        controls = ttk.Frame(main)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Button(controls, text="Compare", command=self.compare).pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=(8, 0))

        self.summary_var = tk.StringVar(value="Paste both texts, then click Compare.")
        ttk.Label(main, textvariable=self.summary_var, style="Summary.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        result_frame = ttk.LabelFrame(main, text="Word differences", padding=8)
        result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        self.result_text = ScrolledText(
            result_frame,
            wrap="word",
            font=("Consolas", 11),
            state="disabled",
        )
        self.result_text.grid(row=0, column=0, sticky="nsew")
        self.result_text.tag_configure("rtl", justify="right", font=("Tahoma", 13))

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.tag_add("rtl", "1.0", "end")
        self.result_text.configure(state="disabled")

    def compare(self) -> None:
        model_output = self.model_text.get("1.0", "end").strip()
        correct_output = self.correct_text.get("1.0", "end").strip()

        if not model_output or not correct_output:
            messagebox.showwarning("Missing text", "Paste both the model output and the correct output.")
            return

        try:
            result = compare_tashkeel(correct_output, model_output)
        except Exception as exc:
            messagebox.showerror("Comparison failed", f"{type(exc).__name__}: {exc}")
            return

        summary = (
            f"Differences: {result.tashkeel_differences} / {result.marked_positions} "
            f"marked positions ({format_rate(result.tashkeel_differences, result.marked_positions)})"
            f"    |    Ignoring final case endings: {result.no_case_differences} / "
            f"{result.no_case_marked_positions} "
            f"({format_rate(result.no_case_differences, result.no_case_marked_positions)})"
        )
        self.summary_var.set(summary)
        self._set_result_text(self._format_details(result))

    def _format_details(self, result: ComparisonResult) -> str:
        lines = [
            f"Tashkeel differences: {result.tashkeel_differences}",
            f"DER-style rate: {format_rate(result.tashkeel_differences, result.marked_positions)}",
            f"Marked positions: {result.marked_positions}",
            f"Matched Arabic letters: {result.matched_letters}",
            f"Correct Arabic letters: {result.correct_letters}",
            f"Model Arabic letters: {result.model_letters}",
            f"Ignoring final case endings: {result.no_case_differences} "
            f"({format_rate(result.no_case_differences, result.no_case_marked_positions)})",
        ]

        alignment_notes = (
            result.base_substitutions
            + result.unmatched_correct_letters
            + result.unmatched_model_letters
        )
        if alignment_notes:
            lines.extend(
                [
                    "",
                    "Base-letter alignment notes (not counted as tashkeel differences):",
                    f"Base substitutions: {result.base_substitutions}",
                    f"Letters only in correct text: {result.unmatched_correct_letters}",
                    f"Letters only in model text: {result.unmatched_model_letters}",
                ]
            )

        lines.append("")
        if not result.word_differences:
            lines.append("No tashkeel differences found.")
            return "\n".join(lines)

        lines.append("Words with tashkeel differences:")
        for index, item in enumerate(result.word_differences, start=1):
            lines.append("")
            lines.append(f"{index}. differences: {item.differences}")
            lines.append(f"correct: {item.correct_word}")
            lines.append(f"model:   {item.model_word}")

        return "\n".join(lines)

    def clear(self) -> None:
        self.model_text.delete("1.0", "end")
        self.correct_text.delete("1.0", "end")
        self.summary_var.set("Paste both texts, then click Compare.")
        self._set_result_text("")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ensure_utf8_console()
    TashkeelCompareUI().run()


if __name__ == "__main__":
    main()
