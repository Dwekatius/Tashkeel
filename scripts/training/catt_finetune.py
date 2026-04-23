"""Fine-tune CATT EO from the released checkpoint on our dataset.

Drop-in replacement for train_catt.py in the abjadai/catt repo.
Place alongside eo_pl.py / tashkeel_tokenizer.py / tashkeel_dataset.py and run.

Differences vs. stock train_catt.py:
  - Loads released weights (best_eo_mlm_ns_epoch_193.pt) before training.
  - Lower LR (overrides the 3e-5 in eo_pl.py's configure_optimizers by
    monkey-patching after construction — keeps us from editing the repo).
  - Fewer epochs (10 is plenty for fine-tuning; stock uses 300 from scratch).
  - RTX 6000 Pro sizing: batch 16, num_workers 8. Adjust for other GPUs.
  - Saves only best-by-val_der to keep disk small.

Usage (from inside the cloned catt repo dir):
    python catt_finetune.py
"""

import os
import glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from tashkeel_dataset import TashkeelDataset, PrePaddingDataLoader
from tashkeel_tokenizer import TashkeelTokenizer
from eo_pl import TashkeelModel

torch.set_float32_matmul_precision("high")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_start_checkpoint() -> str:
    """Pick the best starting weights for a fresh training run.

    Preference order:
      1. external/catt/catt_eo_ft_balanced_v3/catt_eo_ft_best.pt
         (exported state_dict from the current product checkpoint)
      2. external/catt/catt_eo_ft_balanced_v1/catt_eo_ft_best.pt
      3. Latest .ckpt in catt_eo_ft_balanced_v3/ or catt_eo_ft_balanced_v1/
      4. Latest .ckpt in catt_eo_ft_v1/  (stage-1 legacy)
      5. The released pretrained best_eo_mlm_ns_epoch_193.pt

    The script loads via torch.load + load_state_dict (fresh optimizer +
    fresh epoch counter), so either .pt or .ckpt works — we just want the
    best weights to start from.
    """
    catt_root = os.path.join(REPO_ROOT, "external", "catt")
    for folder in ["catt_eo_ft_balanced_v3", "catt_eo_ft_balanced_v1"]:
        balanced_best = os.path.join(catt_root, folder, "catt_eo_ft_best.pt")
        if os.path.exists(balanced_best):
            return balanced_best
    for folder in ["catt_eo_ft_balanced_v3", "catt_eo_ft_balanced_v1", "catt_eo_ft_v1"]:
        matches = glob.glob(os.path.join(catt_root, folder, "catt_eo_ft-epoch=*.ckpt"))
        if matches:
            return max(matches, key=os.path.getmtime)
    return os.path.join(catt_root, "models", "best_eo_mlm_ns_epoch_193.pt")


# ---- config ----
# 5070 Ti local run tuned for throughput without pushing Windows DataLoader too hard.
MODEL_TYPE = "eo"
BATCH_SIZE = 16
NUM_WORKERS = 2                 # Windows DataLoader: keep low to avoid hangs
MAX_SEQ_LEN = 1024
THRESHOLD = 0.6
LR = 1e-5                       # fine-tune LR (stock is 3e-5 from scratch)
MAX_EPOCHS = 5
PRETRAINED_CKPT = find_start_checkpoint()
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0

DATASET_ROOT = os.path.join(REPO_ROOT, "datasets", "arabic_tashkeel", "catt_format_balanced_v3")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")

OUT_DIR = f"catt_{MODEL_TYPE}_ft_balanced_v3"
# ----------------


def main():
    tokenizer = TashkeelTokenizer()

    print(f"Training data root: {DATASET_ROOT}")
    print("Creating Train Dataset...")
    train_ds = TashkeelDataset(TRAIN_DIR, tokenizer, MAX_SEQ_LEN, tashkeel_to_text_ratio_threshold=THRESHOLD)
    train_dl = PrePaddingDataLoader(
        tokenizer,
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )

    print("Creating Val Dataset...")
    val_ds = TashkeelDataset(VAL_DIR, tokenizer, MAX_SEQ_LEN, tashkeel_to_text_ratio_threshold=THRESHOLD)
    val_dl = PrePaddingDataLoader(
        tokenizer,
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )

    print("Building Model...")
    model = TashkeelModel(tokenizer, max_seq_len=MAX_SEQ_LEN, n_layers=6, learnable_pos_emb=False)

    if not os.path.exists(PRETRAINED_CKPT):
        raise FileNotFoundError(
            f"{PRETRAINED_CKPT} not found. Download it first:\n"
            "  mkdir -p models/\n"
            "  wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_eo_mlm_ns_epoch_193.pt"
        )
    print(f"Loading pretrained weights from {PRETRAINED_CKPT} ...")
    state = torch.load(PRETRAINED_CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

    # Override LR to a gentle fine-tune rate without editing eo_pl.py.
    def _configure_optimizers():
        return torch.optim.AdamW(model.parameters(), lr=LR)
    model.configure_optimizers = _configure_optimizers

    ckpt = ModelCheckpoint(
        dirpath=OUT_DIR,
        save_top_k=3,
        save_last=True,
        monitor="val_der",
        mode="min",
        filename=f"catt_{MODEL_TYPE}_ft" + "-{epoch:02d}-{val_loss:.5f}-{val_der:.5f}",
    )

    trainer = Trainer(
        accelerator="cuda",
        devices=-1,
        max_epochs=MAX_EPOCHS,
        callbacks=[TQDMProgressBar(refresh_rate=20), ckpt],
        logger=CSVLogger(save_dir=f"{OUT_DIR}/logs"),
        precision="bf16-mixed",       # RTX 6000 Pro / H100 / A100 only. For older GPUs use "16-mixed".
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"\nDone. Best checkpoints saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
