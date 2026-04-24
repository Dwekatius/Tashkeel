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

## 1. Build The ML Dataset

```powershell
& $py ML_Tashkeel\scripts\build_punctuation_dataset.py
```

Outputs:

- `ML_Tashkeel/data/source/final_tashkeel_train_punct_v4.jsonl`
- `ML_Tashkeel/data/source/final_tashkeel_eval_punct_v4.jsonl`
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

Default base model:

```text
CAMeL-Lab/bert-base-arabic-camelbert-ca
```

Change it with `--model-name`.

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
