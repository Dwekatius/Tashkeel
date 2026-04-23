"""Source-of-truth copy of the punctuation-aware v4 CATT fine-tune script.

Runtime mirror:
  external/catt_punct_v4/catt_finetune.py
"""

from __future__ import annotations

import os
import runpy
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNTIME_SCRIPT = os.path.join(REPO_ROOT, "external", "catt_punct_v4", "catt_finetune.py")
RUNTIME_DIR = os.path.dirname(RUNTIME_SCRIPT)

if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)


if __name__ == "__main__":
    runpy.run_path(RUNTIME_SCRIPT, run_name="__main__")
