"""Fine-tune punctuation-aware CATT EO from the latest local checkpoint.

This is the v4 punctuation-aware training track. It keeps punctuation in the
input stream so the model can use it as context and reproduce it directly in
decoded output. The task is still tashkeel labeling over the input sequence, so
punctuation must already be present in the source text.
"""

from __future__ import annotations

import glob
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from checkpoint_compat import load_state_dict_with_vocab_expansion
from eo_pl import TashkeelModel
from tashkeel_dataset import PrePaddingDataLoader, TashkeelDataset
from tashkeel_tokenizer import TashkeelTokenizer


torch.set_float32_matmul_precision("high")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CATT_ROOT = os.path.abspath(os.path.dirname(__file__))


def find_start_checkpoint() -> str:
    candidates = [
        os.path.join(CATT_ROOT, "catt_eo_ft_balanced_v4_punct", "catt_eo_ft_best.pt"),
        os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v3", "catt_eo_ft_best.pt"),
        os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v1", "catt_eo_ft_best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    for folder in [
        os.path.join(CATT_ROOT, "catt_eo_ft_balanced_v4_punct"),
        os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v3"),
        os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_balanced_v1"),
        os.path.join(REPO_ROOT, "external", "catt", "catt_eo_ft_v1"),
    ]:
        matches = glob.glob(os.path.join(folder, "catt_eo_ft-epoch=*.ckpt"))
        if matches:
            return max(matches, key=os.path.getmtime)

    return os.path.join(REPO_ROOT, "external", "catt", "models", "best_eo_mlm_ns_epoch_193.pt")


MODEL_TYPE = "eo"
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_SEQ_LEN = 1024
THRESHOLD = 0.6
LR = 1e-5
MAX_EPOCHS = 5
PRETRAINED_CKPT = find_start_checkpoint()
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0

DATASET_ROOT = os.path.join(REPO_ROOT, "datasets", "arabic_tashkeel", "catt_format_balanced_v4_punct")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
OUT_DIR = os.path.join(CATT_ROOT, f"catt_{MODEL_TYPE}_ft_balanced_v4_punct")


def trainer_precision() -> str:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"
    return "32-true"


def main():
    tokenizer = TashkeelTokenizer()

    print("Punctuation-aware v4 fine-tune")
    print(f"Training data root: {DATASET_ROOT}")
    print(f"Warm-start checkpoint: {PRETRAINED_CKPT}")

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
        raise FileNotFoundError(f"Start checkpoint not found: {PRETRAINED_CKPT}")

    print(f"Loading warm-start weights from {PRETRAINED_CKPT} ...")
    state = torch.load(PRETRAINED_CKPT, map_location="cpu")
    missing, unexpected = load_state_dict_with_vocab_expansion(model, state)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

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
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[TQDMProgressBar(refresh_rate=20), ckpt],
        logger=CSVLogger(save_dir=os.path.join(OUT_DIR, "logs")),
        precision=trainer_precision(),
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"\nDone. Best checkpoints saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
