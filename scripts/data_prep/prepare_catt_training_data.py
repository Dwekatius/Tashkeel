"""Convert our JSONL splits into CATT's expected folder layout.

CATT's TashkeelDataset reads *.txt files recursively from a folder, one
tashkeel-bearing sentence per line. It filters out:
  - lines <= 5 chars or > max_seq_len (1024)
  - lines with tashkeel/text ratio < 0.6 (configurable via --threshold)
  - everything not Arabic letters / tashkeel (via tokenizer.clean_text)

We emit:
  catt_dataset/train/train.txt
  catt_dataset/val/val.txt
  catt_dataset/test/test.txt       (copy of val so CATT's trainer is happy)

Only the "assistant" field is used — that's the fully-diacritized target.
"""

import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_TRAIN = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_train.jsonl")
SRC_EVAL = os.path.join(ROOT, "datasets", "arabic_tashkeel", "splits", "final_tashkeel_eval.jsonl")
OUT_ROOT = os.path.join(ROOT, "datasets", "arabic_tashkeel", "catt_format")

TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")
# CATT's clean_text keeps: U+0621-U+063A, U+0640-U+0652, U+0670, U+0671,
# plus a few ligatures (FEFB/FEF7/FEF5/FEF9) and space. Everything else is
# replaced with a space. We pre-clean to mirror that so the line count we
# write matches what CATT will actually use, and to avoid double newlines.
KEEP_RE = re.compile(r"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]")


def clean_for_catt(s: str) -> str:
    s = s.replace("ٱ", "ا")
    s = s.replace("\u0640", "")  # tatweel
    s = KEEP_RE.sub(" ", s)
    return " ".join(s.split())


def write_split(jsonl_path: str, out_txt: str):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    kept = 0
    skipped_empty = 0
    skipped_no_tashkeel = 0
    skipped_short = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(out_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("assistant", "")
            cleaned = clean_for_catt(text)
            if not cleaned:
                skipped_empty += 1
                continue
            if not TASHKEEL_RE.search(cleaned):
                skipped_no_tashkeel += 1
                continue
            if len(cleaned) <= 5:
                skipped_short += 1
                continue
            fout.write(cleaned + "\n")
            kept += 1
    return kept, skipped_empty, skipped_no_tashkeel, skipped_short


def main():
    train_txt = os.path.join(OUT_ROOT, "train", "train.txt")
    val_txt = os.path.join(OUT_ROOT, "val", "val.txt")
    test_txt = os.path.join(OUT_ROOT, "test", "test.txt")

    print(f"Writing {train_txt} ...")
    t = write_split(SRC_TRAIN, train_txt)
    print(f"  kept={t[0]}  empty={t[1]}  no_tashkeel={t[2]}  short={t[3]}")

    print(f"Writing {val_txt} ...")
    v = write_split(SRC_EVAL, val_txt)
    print(f"  kept={v[0]}  empty={v[1]}  no_tashkeel={v[2]}  short={v[3]}")

    print(f"Writing {test_txt} (mirror of val) ...")
    with open(val_txt, "r", encoding="utf-8") as f:
        data = f.read()
    os.makedirs(os.path.dirname(test_txt), exist_ok=True)
    with open(test_txt, "w", encoding="utf-8") as f:
        f.write(data)
    print(f"  kept={v[0]}")

    print(f"\nDone. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
