from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CATT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_TYPE = "eo"
TASHKEEL_CODEPOINTS = set(range(0x064B, 0x0653)) | {0x0670}
ARABIC_BASE_RE = re.compile(r"[\u0621-\u064A\u0671]")
NEWLINE_RE = re.compile(r"(\n+)")
TOKEN_RE = re.compile(r"\S+\s*|\s+")
PUNCTUATION_CHUNK_RE = re.compile(r".+?(?:[،؛:,.!?؟]+(?:\s+|$)|$)", re.S)

SAMPLE_TEXT = (
    "قال الطالب ان المشروع انتهى بنجاح، لكننا نحتاج الى واجهة بسيطة للتجربة.\n"
    "اكتب هنا اي نص عربي بدون تشكيل ثم اضغط زر اضافة التشكيل."
)


def ensure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds >= 60:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.2f}s"


def strip_existing_tashkeel(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) not in TASHKEEL_CODEPOINTS)


def is_arabic_base(ch: str) -> bool:
    return bool(ARABIC_BASE_RE.match(ch))


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_BASE_RE.search(text))


def split_predicted_pairs(text: str):
    pairs = []
    base = None
    diac = ""
    for ch in text:
        if ord(ch) in TASHKEEL_CODEPOINTS:
            if base is not None:
                diac += ch
        elif is_arabic_base(ch):
            if base is not None:
                pairs.append((base, diac))
            base = ch
            diac = ""
        else:
            if base is not None:
                pairs.append((base, diac))
                base = None
                diac = ""
    if base is not None:
        pairs.append((base, diac))
    return pairs


def reinsert_punctuation(original_text: str, predicted_text: str):
    predicted_pairs = split_predicted_pairs(predicted_text)
    out = []
    pair_index = 0

    for ch in original_text:
        if ord(ch) in TASHKEEL_CODEPOINTS:
            # Remove user-provided diacritics and replace them with model output.
            continue
        if is_arabic_base(ch):
            out.append(ch)
            if pair_index < len(predicted_pairs):
                _, diac = predicted_pairs[pair_index]
                pair_index += 1
                out.append(diac)
        else:
            out.append(ch)

    leftover_pairs = len(predicted_pairs) - pair_index
    return "".join(out), leftover_pairs


@dataclass
class SegmentTelemetry:
    original_segments: int
    inferred_segments: int
    chunked_segments: int
    warnings: list[str]


class LocalCATTInference:
    def __init__(
        self,
        catt_dir: str = DEFAULT_CATT_DIR,
        checkpoint: str | None = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        self.catt_dir = os.path.abspath(catt_dir)
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.checkpoint = os.path.abspath(checkpoint) if checkpoint else self._find_default_checkpoint()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _find_default_checkpoint(self) -> str:
        preferred_dirs = []
        if self.model_type == "eo":
            preferred_dirs.append(os.path.join(self.catt_dir, "catt_eo_ft_balanced_v3"))
            preferred_dirs.append(os.path.join(self.catt_dir, "catt_eo_ft_balanced_v1"))
        preferred_dirs.append(os.path.join(self.catt_dir, f"catt_{self.model_type}_ft_v1"))

        for ft_dir in preferred_dirs:
            candidates = [
                os.path.join(ft_dir, f"catt_{self.model_type}_ft_best.pt"),
                os.path.join(ft_dir, "last.ckpt"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
            matches = sorted(
                glob.glob(os.path.join(ft_dir, f"catt_{self.model_type}_ft-epoch=*.ckpt")),
                key=os.path.getmtime,
                reverse=True,
            )
            if matches:
                return os.path.abspath(matches[0])
        raise FileNotFoundError(
            "Could not auto-detect a fine-tuned checkpoint under the known EO/ED run folders. "
            "Pass --checkpoint explicitly."
        )

    def load(self):
        if self.model is not None:
            return

        if self.catt_dir not in sys.path:
            sys.path.insert(0, self.catt_dir)

        import torch
        from tashkeel_tokenizer import TashkeelTokenizer

        if self.model_type == "eo":
            from eo_pl import TashkeelModel
            n_layers = 6
        else:
            from ed_pl import TashkeelModel
            n_layers = 3

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        tokenizer = TashkeelTokenizer()
        model = TashkeelModel(
            tokenizer,
            max_seq_len=self.max_seq_len,
            n_layers=n_layers,
            learnable_pos_emb=False,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch. missing={len(missing)} unexpected={len(unexpected)}"
            )

        model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.model = model

    def cleaned_length(self, text: str) -> int:
        cleaned = self.tokenizer.clean_text(strip_existing_tashkeel(text))
        return len(cleaned)

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
            chunks = self._split_long_segment(part)
            if len(chunks) > 1:
                chunked_segments += 1
            for chunk in chunks:
                segments.append(("infer", chunk))
        return segments, original_segments, chunked_segments

    def diacritize(self, text: str):
        self.load()
        t0 = time.time()
        segments, original_segments, chunked_segments = self._prepare_segments(text)
        infer_segments = [segment for kind, segment in segments if kind == "infer"]
        warnings = []

        predictions = []
        for start in range(0, len(infer_segments), self.batch_size):
            batch = infer_segments[start : start + self.batch_size]
            batch_inputs = [strip_existing_tashkeel(segment) for segment in batch]
            outputs = self.model.do_tashkeel_batch(batch_inputs, batch_size=len(batch_inputs), verbose=False)
            predictions.extend(outputs)

        output_parts = []
        pred_index = 0
        for kind, segment in segments:
            if kind == "raw":
                output_parts.append(segment)
                continue
            predicted = predictions[pred_index]
            pred_index += 1
            rebuilt, leftover_pairs = reinsert_punctuation(segment, predicted)
            if leftover_pairs:
                warnings.append(
                    f"Segment {pred_index} left {leftover_pairs} predicted letter(s) unused after punctuation reinsertion."
                )
            output_parts.append(rebuilt)

        result = "".join(output_parts)
        telemetry = {
            "elapsed": time.time() - t0,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "segment_info": SegmentTelemetry(
                original_segments=original_segments,
                inferred_segments=len(infer_segments),
                chunked_segments=chunked_segments,
                warnings=warnings,
            ),
        }
        return result, telemetry


class TashkeelUI:
    def __init__(self, inferencer: LocalCATTInference):
        self.inferencer = inferencer
        self.root = tk.Tk()
        self.root.title("Arabic Tashkeel UI")
        self.root.geometry("1100x820")
        self.root.minsize(920, 700)

        self.event_queue: queue.Queue = queue.Queue()
        self.model_ready = False
        self.busy = False

        self._build_styles()
        self._build_layout()
        self._append_log("Initializing UI.")
        self._set_status("Loading fine-tuned CATT model...")
        self._disable_actions()
        self.root.after(120, self._poll_events)
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _build_styles(self):
        self.root.configure(bg="#efe8db")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#efe8db")
        style.configure("Card.TFrame", background="#fbf7f0")
        style.configure("Title.TLabel", background="#efe8db", foreground="#11332b", font=("Georgia", 22, "bold"))
        style.configure("Sub.TLabel", background="#efe8db", foreground="#36544b", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#fbf7f0", foreground="#0f172a", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", background="#efe8db", foreground="#0f766e", font=("Segoe UI", 10, "bold"))
        style.configure(
            "Accent.TButton",
            background="#0f766e",
            foreground="#ffffff",
            borderwidth=0,
            padding=(14, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", "#115e59"), ("disabled", "#8fbab6")])
        style.configure(
            "Secondary.TButton",
            background="#d97706",
            foreground="#ffffff",
            borderwidth=0,
            padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Secondary.TButton", background=[("active", "#b45309"), ("disabled", "#d8b48a")])
        style.configure(
            "Neutral.TButton",
            background="#e5ddd0",
            foreground="#1f2937",
            borderwidth=0,
            padding=(12, 10),
            font=("Segoe UI", 10),
        )
        style.map("Neutral.TButton", background=[("active", "#d8cfbf"), ("disabled", "#efe8db")])

    def _build_layout(self):
        main = ttk.Frame(self.root, style="Main.TFrame", padding=18)
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main, style="Main.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Arabic Tashkeel", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Paste Arabic text, click one button, and get punctuation-preserved output from your fine-tuned CATT checkpoint.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 10))

        status_row = ttk.Frame(main, style="Main.TFrame")
        status_row.pack(fill="x", pady=(0, 10))
        self.status_var = tk.StringVar(value="Starting up...")
        ttk.Label(status_row, textvariable=self.status_var, style="Status.TLabel").pack(side="left")

        button_row = ttk.Frame(main, style="Main.TFrame")
        button_row.pack(fill="x", pady=(0, 12))
        self.run_button = ttk.Button(
            button_row,
            text="Add Tashkeel",
            style="Accent.TButton",
            command=self._on_run_clicked,
        )
        self.run_button.pack(side="left")
        self.copy_button = ttk.Button(
            button_row,
            text="Copy Output",
            style="Secondary.TButton",
            command=self._copy_output,
        )
        self.copy_button.pack(side="left", padx=(10, 0))
        self.clear_button = ttk.Button(
            button_row,
            text="Clear",
            style="Neutral.TButton",
            command=self._clear_text,
        )
        self.clear_button.pack(side="left", padx=(10, 0))
        self.sample_button = ttk.Button(
            button_row,
            text="Load Sample",
            style="Neutral.TButton",
            command=self._load_sample,
        )
        self.sample_button.pack(side="left", padx=(10, 0))

        panels = ttk.Frame(main, style="Main.TFrame")
        panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=3)
        panels.rowconfigure(1, weight=1)

        input_card = ttk.Frame(panels, style="Card.TFrame", padding=14)
        input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        output_card = ttk.Frame(panels, style="Card.TFrame", padding=14)
        output_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        log_card = ttk.Frame(panels, style="Card.TFrame", padding=14)
        log_card.grid(row=1, column=0, columnspan=2, sticky="nsew")

        ttk.Label(input_card, text="Input", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            input_card,
            text="Best results come from plain Arabic text. Existing diacritics are stripped and replaced.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 8))
        self.input_box = ScrolledText(
            input_card,
            wrap="word",
            font=("Segoe UI", 14),
            undo=True,
            bg="#fffdf8",
            fg="#111827",
            insertbackground="#111827",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.input_box.pack(fill="both", expand=True)
        self.input_box.tag_configure("rtl", justify="right")

        ttk.Label(output_card, text="Output", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            output_card,
            text="Punctuation and line breaks are preserved where possible.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 8))
        self.output_box = ScrolledText(
            output_card,
            wrap="word",
            font=("Segoe UI", 14),
            undo=False,
            bg="#fffcf5",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.output_box.pack(fill="both", expand=True)
        self.output_box.tag_configure("rtl", justify="right")

        ttk.Label(log_card, text="Telemetry", style="CardTitle.TLabel").pack(anchor="w")
        self.log_box = ScrolledText(
            log_card,
            height=10,
            wrap="word",
            font=("Consolas", 10),
            bg="#10211d",
            fg="#e7f7f2",
            insertbackground="#e7f7f2",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.log_box.configure(state="disabled")

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _append_log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _disable_actions(self):
        self.run_button.configure(state="disabled")
        self.copy_button.configure(state="disabled")

    def _enable_actions(self):
        self.run_button.configure(state="normal")
        self.copy_button.configure(state="normal")

    def _set_text(self, widget: ScrolledText, text: str):
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.tag_add("rtl", "1.0", "end")

    def _clear_text(self):
        self._set_text(self.input_box, "")
        self._set_text(self.output_box, "")
        self._append_log("Cleared input and output boxes.")
        self._set_status("Ready.")

    def _load_sample(self):
        self._set_text(self.input_box, SAMPLE_TEXT)
        self._append_log("Loaded sample text.")
        self._set_status("Sample text loaded.")

    def _copy_output(self):
        output = self.output_box.get("1.0", "end").strip()
        if not output:
            self._set_status("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self._append_log("Copied diacritized output to clipboard.")
        self._set_status("Output copied to clipboard.")

    def _load_model_worker(self):
        started = time.time()
        try:
            self.inferencer.load()
            elapsed = time.time() - started
            self.event_queue.put(
                (
                    "model_ready",
                    {
                        "elapsed": elapsed,
                        "device": self.inferencer.device,
                        "checkpoint": self.inferencer.checkpoint,
                        "model_type": self.inferencer.model_type,
                    },
                )
            )
        except Exception:
            self.event_queue.put(("error", traceback.format_exc()))

    def _on_run_clicked(self):
        if self.busy:
            return
        text = self.input_box.get("1.0", "end").rstrip()
        if not text.strip():
            self._set_status("Enter some Arabic text first.")
            return
        self.busy = True
        self.run_button.configure(state="disabled")
        self.copy_button.configure(state="disabled")
        self._set_status("Running model...")
        self._append_log("Starting inference.")
        threading.Thread(target=self._run_inference_worker, args=(text,), daemon=True).start()

    def _run_inference_worker(self, text: str):
        try:
            output, telemetry = self.inferencer.diacritize(text)
            self.event_queue.put(("result", {"output": output, "telemetry": telemetry}))
        except Exception:
            self.event_queue.put(("error", traceback.format_exc()))

    def _poll_events(self):
        try:
            while True:
                event, payload = self.event_queue.get_nowait()
                if event == "model_ready":
                    self.model_ready = True
                    self.busy = False
                    self._enable_actions()
                    self._append_log(
                        f"Model ready in {format_duration(payload['elapsed'])} on {payload['device']}."
                    )
                    self._append_log(f"Model type: {payload['model_type'].upper()}")
                    self._append_log(f"Checkpoint: {payload['checkpoint']}")
                    self._set_status(
                        f"{payload['model_type'].upper()} model ready on {payload['device']}."
                    )
                elif event == "result":
                    self.busy = False
                    self._enable_actions()
                    output = payload["output"]
                    telemetry = payload["telemetry"]
                    segment_info: SegmentTelemetry = telemetry["segment_info"]
                    self._set_text(self.output_box, output)
                    self._append_log(
                        f"Inference finished in {format_duration(telemetry['elapsed'])} on {telemetry['device']}."
                    )
                    self._append_log(
                        f"Segments: original={segment_info.original_segments}, "
                        f"inferred={segment_info.inferred_segments}, "
                        f"chunked={segment_info.chunked_segments}."
                    )
                    for warning in segment_info.warnings:
                        self._append_log(f"Warning: {warning}")
                    self._set_status(
                        f"Done in {format_duration(telemetry['elapsed'])} using {segment_info.inferred_segments} segment(s)."
                    )
                elif event == "error":
                    self.busy = False
                    if self.model_ready:
                        self._enable_actions()
                    self._append_log("Error:\n" + payload)
                    self._set_status("Something went wrong. See telemetry.")
                    messagebox.showerror("Tashkeel UI Error", payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def run(self):
        self.root.mainloop()


def run_cli(args):
    inferencer = LocalCATTInference(
        catt_dir=args.catt_dir,
        checkpoint=args.checkpoint,
        model_type=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    started = time.time()
    inferencer.load()
    load_elapsed = time.time() - started
    print(f"Loaded model in {format_duration(load_elapsed)} on {inferencer.device}")
    print(f"Checkpoint: {inferencer.checkpoint}\n")
    output, telemetry = inferencer.diacritize(args.text)
    print(output)
    segment_info: SegmentTelemetry = telemetry["segment_info"]
    print()
    print(
        f"[telemetry] elapsed={format_duration(telemetry['elapsed'])} "
        f"segments={segment_info.inferred_segments} "
        f"chunked={segment_info.chunked_segments} "
        f"warnings={len(segment_info.warnings)}"
    )
    for warning in segment_info.warnings:
        print(f"[warning] {warning}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to local fine-tuned .ckpt or .pt")
    parser.add_argument("--catt-dir", default=DEFAULT_CATT_DIR, help="Path to external/catt")
    parser.add_argument("--model", choices=["eo", "ed"], default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--text", help="Run once in CLI mode instead of opening the GUI")
    return parser.parse_args()


def main():
    ensure_utf8_console()
    args = parse_args()
    if args.text is not None:
        run_cli(args)
        return

    inferencer = LocalCATTInference(
        catt_dir=args.catt_dir,
        checkpoint=args.checkpoint,
        model_type=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    app = TashkeelUI(inferencer)
    app.run()


if __name__ == "__main__":
    main()
