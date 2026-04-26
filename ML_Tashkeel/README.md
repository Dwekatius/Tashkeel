# ML_Tashkeel

Self-contained ML workspace for the punctuation-restoration stage.

Goal:

```text
plain Arabic text -> punctuation restorer -> v4 CATT tashkeel model -> final text
```

This folder intentionally copies and derives its own data. The original project
datasets are read-only inputs.

## PowerShell Setup

```powershell
$py = "C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
```

## Quick Punctuation UI

Use this for fast manual testing without running the full tashkeel pipeline:

```powershell
& $py debug_tools\punctuation_web_ui.py
```

Or:

```powershell
.\debug_tools\launchers\run_punctuation_ui.ps1
```

Default URL:

```text
http://127.0.0.1:8776/
```

The UI includes presets for:

- official baseline, calibrated
- official baseline, raw
- balanced own_v2, raw
- balanced own_v2, tuned

It is intentionally punctuation-only. The main tashkeel UI remains separate.

## 1. Build The ML Dataset

```powershell
& $py ML_Tashkeel\scripts\build_punctuation_dataset.py
```

Outputs:

- `ML_Tashkeel/data/source/final_tashkeel_train.jsonl`
- `ML_Tashkeel/data/source/final_tashkeel_eval.jsonl`
- `ML_Tashkeel/data/processed/punctuation_train.jsonl`
- `ML_Tashkeel/data/processed/punctuation_eval.jsonl`
- `ML_Tashkeel/reports/dataset_stats.json`

Each processed row contains word tokens and one punctuation label after each
word:

```text
NONE / COMMA / PERIOD / QUESTION / COLON / SEMICOLON / EXCLAMATION
```

## 2. Quick Smoke Training

Use this first to verify the full training/eval path before launching the real
run. By default the trainer shows one filling progress bar with step count,
elapsed time, ETA, loss, learning rate, gradient norm, and latest eval metrics.

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --max-train-examples 2000 `
  --max-eval-examples 500 `
  --epochs 1 `
  --eval-steps 50 `
  --save-steps 50 `
  --logging-steps 10 `
  --batch-size 8 `
  --eval-batch-size 16 `
  --bf16
```

Expected telemetry shape:

```text
training:  42%|████...| 6900/16371 [.., elapsed=1h 10m 4s, eta=1h 36m 2s, loss=..., lr=..., grad=..., eval_loss=..., acc=..., punct_f1=...]
```

Use `--no-progress-bar` if you want compact line logs instead.

## 3. Full Training

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --epochs 3 `
  --batch-size 32 `
  --eval-batch-size 32 `
  --learning-rate 2e-5 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

With the current processed dataset, `--batch-size 32` means about `5,457`
optimizer steps per epoch, or `16,371` total steps for 3 epochs.

## 3b. Build And Train Augmented v2

The v2 augmentation keeps the original processed training rows unchanged and
adds targeted copies for weak punctuation labels.

```powershell
& $py ML_Tashkeel\scripts\build_augmented_punctuation_dataset.py
```

Outputs:

- `ML_Tashkeel/data/processed/punctuation_train_aug_v2.jsonl`
- `ML_Tashkeel/reports/augmentation_v2_report.json`

Train the v2 comparison model:

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --train-file ML_Tashkeel\data\processed\punctuation_train_aug_v2.jsonl `
  --output-dir ML_Tashkeel\models\punctuation_restorer_aug_v2_bs48 `
  --epochs 3 `
  --batch-size 48 `
  --eval-batch-size 48 `
  --learning-rate 2e-5 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

Current result: this trained successfully, but it is not the best model.
`punctuation_restorer_aug_v2_bs48` trails the calibrated baseline on official
eval and on `docs/qa/punc_tests.txt`, so keep it as an experiment rather than promoting
it.

Default base model:

```text
CAMeL-Lab/bert-base-arabic-camelbert-ca
```

Change it with `--model-name`.

## 3c. Build And Train Own Dataset v1

The local CSV was moved and cleaned into:

- `datasets/arabic_tashkeel/raw/my_own_dataset_original.csv`
- `datasets/arabic_tashkeel/raw/my_own_dataset_clean.csv`

Rebuild the versioned encoder splits:

```powershell
& $py scripts\data_prep\prepare_own_dataset_v1.py
```

Build the ML punctuation files from those combined splits:

```powershell
& $py ML_Tashkeel\scripts\build_punctuation_dataset.py `
  --train-source datasets\arabic_tashkeel\splits\final_tashkeel_train_own_v1.jsonl `
  --eval-source datasets\arabic_tashkeel\splits\final_tashkeel_eval_own_v1.jsonl `
  --train-output-name punctuation_train_own_v1.jsonl `
  --eval-output-name punctuation_eval_own_v1.jsonl `
  --stats-output-name dataset_stats_own_v1.json
```

Build the own-data plus targeted-augmentation train file:

```powershell
& $py ML_Tashkeel\scripts\build_augmented_punctuation_dataset.py `
  --train-file ML_Tashkeel\data\processed\punctuation_train_own_v1.jsonl `
  --output-file ML_Tashkeel\data\processed\punctuation_train_own_aug_v2.jsonl `
  --report-name augmentation_own_v1_aug_v2_report.json
```

Train the real-data-only own comparison model first:

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --train-file ML_Tashkeel\data\processed\punctuation_train_own_v1.jsonl `
  --eval-file ML_Tashkeel\data\processed\punctuation_eval_own_v1.jsonl `
  --output-dir ML_Tashkeel\models\punctuation_restorer_own_v1_bs48 `
  --epochs 3 `
  --batch-size 48 `
  --eval-batch-size 48 `
  --learning-rate 2e-5 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

Current result: this trained successfully, but it is also not the best model.
Tuned official eval reached `punct_micro_f1=0.7898`, below the original
calibrated baseline at `0.8167`. On `docs/qa/punc_tests.txt`, tuned `own_v1` reached
`punct_micro_f1=0.6176` with `5/140` exact rows, which is a small manual-QA
improvement but not enough to replace the baseline.

Only train the own-data plus synthetic augmentation comparison after the
real-data-only run has been evaluated:

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --train-file ML_Tashkeel\data\processed\punctuation_train_own_aug_v2.jsonl `
  --eval-file ML_Tashkeel\data\processed\punctuation_eval_own_v1.jsonl `
  --output-dir ML_Tashkeel\models\punctuation_restorer_own_aug_v2_bs48 `
  --epochs 3 `
  --batch-size 48 `
  --eval-batch-size 48 `
  --learning-rate 2e-5 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

## 3d. Build Balanced Own v2

This is the next preferred dataset experiment. It uses real rows only:

- baseline train rows stay unchanged
- `own_dataset_v1` rows are repeated
- rows with real `QUESTION`, `COLON`, `SEMICOLON`, and `EXCLAMATION` labels are oversampled
- no punctuation labels are invented or rewritten

Build it:

```powershell
& $py ML_Tashkeel\scripts\build_balanced_own_punctuation_dataset.py
```

Outputs:

- `ML_Tashkeel/data/processed/punctuation_train_own_v2_balanced.jsonl`
- `ML_Tashkeel/reports/balanced_own_v2_report.json`

Current shape:

```text
rows:      190,129
chunks:    190,227
batch 32:  5,945 steps/epoch
batch 48:  3,964 steps/epoch
```

Recommended first training run:

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --train-file ML_Tashkeel\data\processed\punctuation_train_own_v2_balanced.jsonl `
  --eval-file ML_Tashkeel\data\processed\punctuation_eval_own_v1.jsonl `
  --output-dir ML_Tashkeel\models\punctuation_restorer_own_v2_balanced_bs32_cw8 `
  --epochs 3 `
  --batch-size 32 `
  --eval-batch-size 48 `
  --learning-rate 2e-5 `
  --max-class-weight 8 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

Current result: this is the strongest comparison so far, but still not a clean
replacement. Tuned official eval reached `punct_micro_f1=0.8141`, very close to
the original calibrated baseline at `0.8167`. On `docs/qa/punc_tests.txt`, raw balanced
v2 reached `punct_micro_f1=0.6433` with `9/140` exact rows, better than the
baseline raw `0.6083` with `4/140` exact rows.

Best balanced-v2 thresholds on official eval:

```text
COMMA=0.95,PERIOD=0.5,QUESTION=0.97,COLON=0.97,SEMICOLON=0.9,EXCLAMATION=0.99
```

Keep the original calibrated baseline as the official best for now, but use
balanced v2 as the main side-by-side QA candidate.

## 3e. Continue From Baseline On Balanced v2

This gently fine-tunes the current best checkpoint for one epoch on the balanced
real-data dataset.

```powershell
& $py ML_Tashkeel\scripts\train_punctuation_restorer.py `
  --model-name ML_Tashkeel\models\punctuation_restorer `
  --train-file ML_Tashkeel\data\processed\punctuation_train_own_v2_balanced.jsonl `
  --eval-file ML_Tashkeel\data\processed\punctuation_eval.jsonl `
  --output-dir ML_Tashkeel\models\punctuation_restorer_baseline_ft_own_v2_bs32_cw8_lr5e6 `
  --epochs 1 `
  --batch-size 32 `
  --eval-batch-size 48 `
  --learning-rate 5e-6 `
  --max-class-weight 8 `
  --eval-steps 500 `
  --save-steps 500 `
  --logging-steps 50 `
  --bf16
```

Current result: not promoted. Tuned official eval reached
`punct_micro_f1=0.8154`, very close to the baseline `0.8167`, but manual
`docs/qa/punc_tests.txt` raw score was `0.6259`, below raw balanced v2 at `0.6433`.

Build CATT encoder-format data from the same own-v1 combined splits:

```powershell
& $py scripts\data_prep\prepare_balanced_catt_training_data_v4_punct.py `
  --train-source datasets\arabic_tashkeel\splits\final_tashkeel_train_own_v1.jsonl `
  --eval-source datasets\arabic_tashkeel\splits\final_tashkeel_eval_own_v1.jsonl `
  --out-root datasets\arabic_tashkeel\catt_format_own_v1_punct
```

## 4. Evaluate

```powershell
& $py ML_Tashkeel\scripts\eval_punctuation_restorer.py `
  --model-dir ML_Tashkeel\models\punctuation_restorer
```

The evaluator prints telemetry and writes:

- `ML_Tashkeel/reports/punctuation_eval_report.json`
- `ML_Tashkeel/reports/punctuation_confusion_matrix.csv`

If the model over-punctuates, tune per-label thresholds before spending another
full training run.

## 4b. Tune Inference Thresholds

This keeps the trained model fixed and searches per-label confidence thresholds.

```powershell
& $py ML_Tashkeel\scripts\tune_punctuation_thresholds.py `
  --model-dir ML_Tashkeel\models\punctuation_restorer `
  --batch-size 64
```

Best current eval-optimized thresholds:

```text
COMMA=0.95,QUESTION=0.9,COLON=0.98,SEMICOLON=0.93,EXCLAMATION=0.98
```

Use thresholds during punctuation-only inference:

```powershell
& $py ML_Tashkeel\scripts\punctuate_text.py `
  --model-dir ML_Tashkeel\models\punctuation_restorer `
  --label-thresholds "COMMA=0.95,QUESTION=0.9,COLON=0.98,SEMICOLON=0.93,EXCLAMATION=0.98" `
  --text "قالت ليلى هل عاد سامر"
```

## 5. Try Punctuation Only

```powershell
& $py ML_Tashkeel\scripts\punctuate_text.py `
  --model-dir ML_Tashkeel\models\punctuation_restorer `
  --text "قالت ليلى هل عاد سامر"
```

## 6. Full Pipeline

Runs punctuation restoration first, then sends the punctuated text to the
existing v4 CATT tashkeel backend.

```powershell
& $py ML_Tashkeel\scripts\run_punctuation_tashkeel_pipeline.py `
  --punct-model-dir ML_Tashkeel\models\punctuation_restorer `
  --text "قالت ليلى هل عاد سامر"
```
