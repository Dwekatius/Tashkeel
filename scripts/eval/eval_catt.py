"""Evaluate CATT on our final_tashkeel_eval.jsonl with console telemetry.

Features:
  - Auto-detects the local fine-tuned checkpoint when available.
  - Falls back to the pip-installed catt_tashkeel package if requested.
  - Prints run configuration, source counts, progress telemetry, and warnings.
  - Surfaces inference failures and alignment failures with example rows.

Examples:
    python tests/eval_catt.py
    python tests/eval_catt.py --use-package
    python tests/eval_catt.py --checkpoint external/catt/catt_eo_ft_v1/catt_eo_ft_best.pt
    python tests/eval_catt.py --per-source 50 --batch 32 --show-failures 8
    python tests/eval_catt.py --full --max-len 1024
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "arabic_tashkeel",
    "splits",
    "final_tashkeel_eval.jsonl",
)
DEFAULT_CATT_DIR = os.path.join(PROJECT_ROOT, "external", "catt")
SEED = 42

TASHKEEL_CODEPOINTS = set(range(0x064B, 0x0653)) | {0x0670}
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")


def ensure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def pct(num: int, den: int) -> str:
    return f"{(num / den * 100):6.2f}%" if den else "   n/a"


def snippet(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize_counter(counter: Counter, top_k: int = 4) -> str:
    if not counter:
        return "n/a"
    items = counter.most_common(top_k)
    parts = [f"{key}:{value}" for key, value in items]
    remaining = len(counter) - len(items)
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return ", ".join(parts)


def is_arabic_letter(ch: str) -> bool:
    return 0x0621 <= ord(ch) <= 0x064A


def normalize_for_alignment(text: str) -> str:
    # Match CATT tokenizer normalization so orthographic variants do not
    # create fake alignment errors, especially Quran rows with alif wasla.
    return text.replace("\u0671", "\u0627").replace("\u0640", "")


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text)


def split_pairs(text: str):
    """[(base_char, diacritics_string), ...] for Arabic letters only."""
    text = normalize_for_alignment(text)
    pairs = []
    base = None
    diac = ""
    for ch in text:
        if ord(ch) in TASHKEEL_CODEPOINTS:
            if base is not None:
                diac += ch
        elif is_arabic_letter(ch):
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


def normalize_diac(diac: str) -> str:
    return "".join(sorted(diac))


def strip_case_endings(text: str) -> str:
    words = text.split(" ")
    out_words = []
    for word in words:
        if not word:
            out_words.append(word)
            continue
        pairs = split_pairs(word)
        if not pairs:
            out_words.append(word)
            continue
        last_base, _ = pairs[-1]
        pairs[-1] = (last_base, "")
        out_words.append("".join(base + diac for base, diac in pairs))
    return " ".join(out_words)


def compute_der(ref: str, pred: str):
    """Returns (errors, total_positions, aligned?)."""
    ref_pairs = split_pairs(ref)
    pred_pairs = split_pairs(pred)
    if len(ref_pairs) != len(pred_pairs):
        return None, None, False
    errors = 0
    total = 0
    for (ref_base, ref_diac), (pred_base, pred_diac) in zip(ref_pairs, pred_pairs):
        if ref_base != pred_base:
            return None, None, False
        if ref_diac or pred_diac:
            total += 1
            if normalize_diac(ref_diac) != normalize_diac(pred_diac):
                errors += 1
    return errors, total, True


def compute_wer(ref: str, pred: str):
    ref_words = ref.split()
    pred_words = pred.split()
    if len(ref_words) != len(pred_words):
        return None, len(ref_words), False
    errors = sum(1 for a, b in zip(ref_words, pred_words) if a != b)
    return errors, len(ref_words), True


def analyze_alignment(ref: str, pred: str):
    ref_pairs = split_pairs(ref)
    pred_pairs = split_pairs(pred)
    if len(ref_pairs) != len(pred_pairs):
        return {
            "category": "letter_count_mismatch",
            "detail": f"reference has {len(ref_pairs)} Arabic letters, prediction has {len(pred_pairs)}",
        }
    for idx, ((ref_base, _), (pred_base, _)) in enumerate(zip(ref_pairs, pred_pairs)):
        if ref_base != pred_base:
            return {
                "category": "base_char_mismatch",
                "detail": f"first differing base char at position {idx}: ref={ref_base!r}, pred={pred_base!r}",
            }
    return {"category": "unknown_alignment_issue", "detail": "alignment failed for an unknown reason"}


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_rows(rows, per_source: int, full: bool):
    if full:
        return list(rows)
    random.seed(SEED)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["source"]].append(row)
    sampled = []
    for source, items in buckets.items():
        random.shuffle(items)
        sampled.extend(items[:per_source])
    random.shuffle(sampled)
    return sampled


def print_source_counts(label: str, rows):
    counts = Counter(row["source"] for row in rows)
    print(label)
    for source in sorted(counts):
        print(f"  {source:<22} {counts[source]:>5}")
    print(f"  {'TOTAL':<22} {sum(counts.values()):>5}\n")


def extract_val_der_from_name(path: str) -> float | None:
    match = re.search(r"val_der=([0-9.]+)", os.path.basename(path))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def find_default_checkpoint(model: str, catt_dir: str) -> str | None:
    preferred_dirs = []
    if model == "eo":
        preferred_dirs.append(os.path.join(catt_dir, "catt_eo_ft_balanced_v3"))
        preferred_dirs.append(os.path.join(catt_dir, "catt_eo_ft_balanced_v1"))
    preferred_dirs.append(os.path.join(catt_dir, f"catt_{model}_ft_v1"))

    for ft_dir in preferred_dirs:
        pt_candidate = os.path.join(ft_dir, f"catt_{model}_ft_best.pt")
        if os.path.exists(pt_candidate):
            return pt_candidate

        matches = glob.glob(os.path.join(ft_dir, f"catt_{model}_ft-epoch=*.ckpt"))
        if matches:
            scored = []
            for path in matches:
                val_der = extract_val_der_from_name(path)
                scored.append((float("inf") if val_der is None else val_der, -os.path.getmtime(path), path))
            scored.sort()
            return scored[0][2]

        last_candidate = os.path.join(ft_dir, "last.ckpt")
        if os.path.exists(last_candidate):
            return last_candidate

        matches = sorted(
            glob.glob(os.path.join(ft_dir, f"catt_{model}_ft-epoch=*.ckpt")),
            key=os.path.getmtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    return None


def resolve_checkpoint(args) -> str | None:
    if args.use_package:
        return None

    if args.checkpoint:
        checkpoint = os.path.abspath(args.checkpoint)
        if not os.path.exists(checkpoint):
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = find_default_checkpoint(args.model, args.catt_dir)
    if checkpoint:
        print(f"Auto-detected local checkpoint: {checkpoint}")
    else:
        print("No local fine-tuned checkpoint found. Falling back to pip package model.")
    return checkpoint


def load_model(args):
    if not args.checkpoint:
        print(f"Loading CATT model from pip package: {args.model.upper()} ...")
        try:
            from catt_tashkeel import CATTEncoderDecoder, CATTEncoderOnly
        except ImportError as exc:
            raise SystemExit(
                "Could not import catt_tashkeel. Install it, or run with --checkpoint "
                "in the env that has torch + the local external/catt repo."
            ) from exc
        model = (CATTEncoderOnly if args.model == "eo" else CATTEncoderDecoder)()
        print("Model loaded from pip package.\n")
        return model

    catt_dir = os.path.abspath(args.catt_dir)
    if not os.path.isdir(catt_dir):
        raise SystemExit(f"CATT repo path does not exist: {catt_dir}")
    if catt_dir not in sys.path:
        sys.path.insert(0, catt_dir)

    try:
        import torch
        from tashkeel_tokenizer import TashkeelTokenizer
        n_layers = 6 if args.model == "eo" else 3
        if args.model == "eo":
            from eo_pl import TashkeelModel
        else:
            from ed_pl import TashkeelModel
    except ImportError as exc:
        raise SystemExit(
            "Failed to import local CATT runtime. Use the training env / unsloth env, "
            "or pass --use-package to evaluate the pip-installed ONNX model."
        ) from exc

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"Loading local CATT checkpoint: {args.checkpoint}")
    print(f"CATT source dir: {catt_dir}")

    tokenizer = TashkeelTokenizer()
    model = TashkeelModel(tokenizer, max_seq_len=1024, n_layers=n_layers, learnable_pos_emb=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded. missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    print(f"Using device: {device}\n")
    return model


def run_inference(records, model, args):
    total_rows = len(records)
    total_batches = (total_rows + args.batch - 1) // args.batch
    start_time = time.time()
    batch_failures = 0
    inference_failure_examples = []

    for batch_num, batch_start in enumerate(range(0, total_rows, args.batch), start=1):
        batch_records = records[batch_start : batch_start + args.batch]
        batch_inputs = [record["plain"] for record in batch_records]
        batch_t0 = time.time()
        batch_sources = Counter(record["source"] for record in batch_records)
        batch_char_lengths = [len(record["ref"]) for record in batch_records]

        try:
            outputs = model.do_tashkeel_batch(batch_inputs, batch_size=args.batch, verbose=False)
            if len(outputs) != len(batch_inputs):
                raise RuntimeError(
                    f"model returned {len(outputs)} outputs for {len(batch_inputs)} inputs"
                )
            for record, pred in zip(batch_records, outputs):
                record["pred"] = pred
        except Exception as exc:
            batch_failures += 1
            print(
                f"[error] Batch {batch_num}/{total_batches} failed "
                f"({type(exc).__name__}: {exc}). Retrying rows one by one..."
            )
            for record in batch_records:
                try:
                    outputs = model.do_tashkeel_batch([record["plain"]], batch_size=1, verbose=False)
                    if len(outputs) != 1:
                        raise RuntimeError(f"model returned {len(outputs)} outputs for a single row")
                    record["pred"] = outputs[0]
                except Exception as row_exc:
                    record["infer_error"] = f"{type(row_exc).__name__}: {row_exc}"
                    if len(inference_failure_examples) < args.show_failures:
                        inference_failure_examples.append(
                            {
                                "source": record["source"],
                                "chars": len(record["ref"]),
                                "plain": snippet(record["plain"]),
                                "error": record["infer_error"],
                            }
                        )
                    print(
                        f"        row failed source={record['source']} chars={len(record['ref'])} "
                        f"error={record['infer_error']}"
                    )

        done = min(batch_start + len(batch_records), total_rows)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total_rows - done) / rate if rate > 0 else 0.0
        batch_elapsed = time.time() - batch_t0
        should_log = (
            batch_num == 1
            or batch_num == total_batches
            or batch_num % args.progress_every == 0
        )
        if should_log:
            print(
                f"[infer] batch {batch_num:>4}/{total_batches:<4}  "
                f"rows {done:>5}/{total_rows:<5}  "
                f"{(done / total_rows * 100):5.1f}%  "
                f"avg {rate:7.1f} rows/s  "
                f"last {batch_elapsed:5.2f}s  "
                f"ETA {format_duration(eta)}"
            )
            print(
                f"        types {summarize_counter(batch_sources)}  "
                f"chars min/avg/max "
                f"{min(batch_char_lengths)}/{sum(batch_char_lengths) // len(batch_char_lengths)}/{max(batch_char_lengths)}"
            )

    total_time = time.time() - start_time
    total_failures = sum(1 for record in records if record["infer_error"])
    print(
        f"\nInference finished in {format_duration(total_time)} "
        f"({total_rows / total_time:.1f} rows/s)."
    )
    print(f"Batch-level inference retries: {batch_failures}")
    print(f"Row-level inference failures: {total_failures}\n")

    return {
        "total_time": total_time,
        "batch_failures": batch_failures,
        "row_failures": total_failures,
        "failure_examples": inference_failure_examples,
    }


def evaluate_records(records, args):
    agg = defaultdict(
        lambda: {
            "n": 0,
            "aligned": 0,
            "inference_fail": 0,
            "misaligned": 0,
            "alignment_reasons": Counter(),
            "der_err": 0,
            "der_tot": 0,
            "der_nce_err": 0,
            "der_nce_tot": 0,
            "wer_err": 0,
            "wer_tot": 0,
            "wer_nce_err": 0,
            "wer_nce_tot": 0,
        }
    )
    totals = defaultdict(int)
    misalignment_examples = []
    worst_rows = []

    for record in records:
        source = record["source"]
        ref = record["ref"]
        pred = record["pred"]
        agg[source]["n"] += 1

        if record["infer_error"]:
            agg[source]["inference_fail"] += 1
            continue

        der_err, der_tot, aligned = compute_der(ref, pred)
        if not aligned:
            agg[source]["misaligned"] += 1
            info = analyze_alignment(ref, pred)
            agg[source]["alignment_reasons"][info["category"]] += 1
            if len(misalignment_examples) < args.show_failures:
                misalignment_examples.append(
                    {
                        "source": source,
                        "reason": info["category"],
                        "detail": info["detail"],
                        "ref": snippet(ref),
                        "pred": snippet(pred),
                    }
                )
            continue

        agg[source]["aligned"] += 1
        agg[source]["der_err"] += der_err
        agg[source]["der_tot"] += der_tot

        wer_err, wer_tot, wer_ok = compute_wer(ref, pred)
        if wer_ok:
            agg[source]["wer_err"] += wer_err
            agg[source]["wer_tot"] += wer_tot

        ref_nce = strip_case_endings(ref)
        pred_nce = strip_case_endings(pred)
        der_nce_err, der_nce_tot, _ = compute_der(ref_nce, pred_nce)
        wer_nce_err, wer_nce_tot, wer_nce_ok = compute_wer(ref_nce, pred_nce)

        if der_nce_err is not None:
            agg[source]["der_nce_err"] += der_nce_err
            agg[source]["der_nce_tot"] += der_nce_tot
        if wer_nce_ok:
            agg[source]["wer_nce_err"] += wer_nce_err
            agg[source]["wer_nce_tot"] += wer_nce_tot

        row_der = (der_err / der_tot * 100) if der_tot else 0.0
        row_wer = (wer_err / wer_tot * 100) if wer_ok and wer_tot else None
        worst_rows.append(
            {
                "source": source,
                "row_der": row_der,
                "row_wer": row_wer,
                "ref": snippet(ref),
                "pred": snippet(pred),
            }
        )

    for source_data in agg.values():
        for key, value in source_data.items():
            if key == "alignment_reasons":
                continue
            totals[key] += value

    worst_rows.sort(key=lambda item: (item["row_der"], item["row_wer"] or -1), reverse=True)
    worst_rows = worst_rows[: args.show_worst]

    return agg, totals, misalignment_examples, worst_rows


def print_report(args, rows, agg, totals, inference_stats, misalignment_examples, worst_rows):
    print("=" * 95)
    print(f"CATT {args.model.upper()} Evaluation - {os.path.basename(EVAL_PATH)}")
    print("=" * 95)

    header = (
        f"{'source':<22}{'rows':>6}{'align%':>8}{'DER':>10}"
        f"{'DER-noCE':>11}{'WER':>10}{'WER-noCE':>11}"
    )
    print(header)
    print("-" * len(header))

    for source in sorted(agg):
        source_data = agg[source]
        align_pct = source_data["aligned"] / source_data["n"] * 100 if source_data["n"] else 0.0
        if source_data["aligned"] == 0:
            print(
                f"{source:<22}{source_data['n']:>6}{align_pct:>7.1f}%"
                f"{'n/a':>10}{'n/a':>11}{'n/a':>10}{'n/a':>11}"
            )
            continue
        print(
            f"{source:<22}{source_data['n']:>6}{align_pct:>7.1f}%"
            f"{pct(source_data['der_err'], source_data['der_tot']):>10}"
            f"{pct(source_data['der_nce_err'], source_data['der_nce_tot']):>11}"
            f"{pct(source_data['wer_err'], source_data['wer_tot']):>10}"
            f"{pct(source_data['wer_nce_err'], source_data['wer_nce_tot']):>11}"
        )

    print("-" * len(header))
    overall_align = totals["aligned"] / totals["n"] * 100 if totals["n"] else 0.0
    print(
        f"{'OVERALL':<22}{totals['n']:>6}{overall_align:>7.1f}%"
        f"{pct(totals['der_err'], totals['der_tot']):>10}"
        f"{pct(totals['der_nce_err'], totals['der_nce_tot']):>11}"
        f"{pct(totals['wer_err'], totals['wer_tot']):>10}"
        f"{pct(totals['wer_nce_err'], totals['wer_nce_tot']):>11}"
    )
    print()

    print("Run Summary:")
    print(f"  rows loaded:              {len(rows)}")
    print(f"  rows aligned:             {totals['aligned']}")
    print(f"  alignment failures:       {totals['misaligned']}")
    print(f"  inference failures:       {totals['inference_fail']}")
    print(f"  total inference time:     {format_duration(inference_stats['total_time'])}")
    print(f"  average throughput:       {len(rows) / inference_stats['total_time']:.1f} rows/s")
    print()

    worst_sources = sorted(
        (
            (
                source,
                agg[source]["der_err"] / agg[source]["der_tot"] * 100 if agg[source]["der_tot"] else None,
                agg[source]["aligned"],
                agg[source]["misaligned"],
            )
            for source in agg
        ),
        key=lambda item: (-1 if item[1] is None else item[1]),
        reverse=True,
    )
    print("Worst Sources By DER:")
    for source, source_der, aligned, misaligned in worst_sources[:5]:
        der_text = f"{source_der:.2f}%" if source_der is not None else "n/a"
        print(
            f"  {source:<22} DER {der_text:>8}  "
            f"aligned={aligned:>4}  misaligned={misaligned:>4}"
        )
    print()

    if inference_stats["failure_examples"]:
        print("Inference Failure Examples:")
        for idx, example in enumerate(inference_stats["failure_examples"], start=1):
            print(
                f"  {idx}. source={example['source']} chars={example['chars']} "
                f"error={example['error']}"
            )
            print(f"     plain: {example['plain']}")
        print()

    if misalignment_examples:
        print("Alignment Failure Examples:")
        for idx, example in enumerate(misalignment_examples, start=1):
            print(
                f"  {idx}. source={example['source']} reason={example['reason']} "
                f"detail={example['detail']}"
            )
            print(f"     ref : {example['ref']}")
            print(f"     pred: {example['pred']}")
        print()

    if worst_rows:
        print("Worst Aligned Rows By DER:")
        for idx, row in enumerate(worst_rows, start=1):
            wer_text = "n/a" if row["row_wer"] is None else f"{row['row_wer']:.2f}%"
            print(
                f"  {idx}. source={row['source']} row_DER={row['row_der']:.2f}% "
                f"row_WER={wer_text}"
            )
            print(f"     ref : {row['ref']}")
            print(f"     pred: {row['pred']}")
        print()

    print("Notes:")
    print("  align% = fraction of rows where predicted Arabic letter sequence matched reference.")
    print("  DER    = diacritic error rate on aligned rows, including case endings.")
    print("  WER    = word error rate on aligned rows, where any wrong diacritic makes the word wrong.")
    print("  DER-noCE / WER-noCE ignore final-letter case endings.")


def main():
    ensure_utf8_console()

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Use all 16,220 rows")
    parser.add_argument("--model", choices=["eo", "ed"], default="eo")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--max-len",
        type=int,
        default=500,
        help="Skip rows longer than this many chars to stay comparable with the baseline run",
    )
    parser.add_argument("--checkpoint", help="Path to local fine-tuned .ckpt or .pt")
    parser.add_argument(
        "--catt-dir",
        default=DEFAULT_CATT_DIR,
        help="Path to the local external/catt repo used when loading a local checkpoint",
    )
    parser.add_argument("--use-package", action="store_true", help="Force use of pip-installed catt_tashkeel")
    parser.add_argument("--progress-every", type=int, default=10, help="Log inference telemetry every N batches")
    parser.add_argument("--show-failures", type=int, default=5, help="Show up to N failure examples")
    parser.add_argument("--show-worst", type=int, default=5, help="Show up to N worst aligned rows")
    args = parser.parse_args()

    args.checkpoint = resolve_checkpoint(args)

    print("Run Configuration:")
    print(f"  python executable:       {sys.executable}")
    print(f"  working directory:       {os.getcwd()}")
    print(f"  eval dataset:            {EVAL_PATH}")
    print(f"  model type:              {args.model.upper()}")
    print(f"  batch size:              {args.batch}")
    print(f"  max length:              {args.max_len}")
    print(f"  sample mode:             {'full set' if args.full else f'{args.per_source} per source'}")
    print(f"  model source:            {'pip package' if not args.checkpoint else 'local checkpoint'}")
    if args.checkpoint:
        print(f"  checkpoint:              {args.checkpoint}")
    print()

    model = load_model(args)

    print(f"Loading eval rows from {os.path.basename(EVAL_PATH)} ...")
    all_rows = load_rows(EVAL_PATH)
    print(f"Loaded {len(all_rows)} raw rows.\n")
    print_source_counts("Raw Source Counts:", all_rows)

    rows = sample_rows(all_rows, args.per_source, args.full)
    print_source_counts("Sampled Source Counts:", rows)

    skipped_long = [row for row in rows if len(row["assistant"]) > args.max_len]
    if skipped_long:
        print(f"Skipping {len(skipped_long)} rows longer than {args.max_len} chars.")
        print_source_counts("Skipped-Long Source Counts:", skipped_long)
    rows = [row for row in rows if len(row["assistant"]) <= args.max_len]

    records = [
        {
            "source": row["source"],
            "ref": row["assistant"],
            "plain": strip_tashkeel(row["assistant"]).strip(),
            "pred": None,
            "infer_error": None,
        }
        for row in rows
    ]

    inference_stats = run_inference(records, model, args)
    agg, totals, misalignment_examples, worst_rows = evaluate_records(records, args)
    print_report(args, rows, agg, totals, inference_stats, misalignment_examples, worst_rows)


if __name__ == "__main__":
    main()
