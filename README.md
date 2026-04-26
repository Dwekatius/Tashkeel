# Tashkeel

Arabic diacritization project built around CATT EO.

Read [handover.txt](handover.txt) first for the current project state. That file
is the source of truth when older experiment notes disagree.

## Current Runtime

The main browser UI uses the exported v4 punctuation-aware CATT checkpoint when
it exists:

```text
external/catt_punct_v4/catt_eo_ft_balanced_v4_punct/catt_eo_ft_best.pt
```

v4 preserves punctuation already present in the input. It does not generate
missing punctuation from plain text, so punctuation restoration is a separate ML
stage in `ML_Tashkeel/`.

## Useful Commands

Use the Unsloth Studio Python environment:

```powershell
$py = "C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
```

Run the main tashkeel web UI:

```powershell
& $py debug_tools\tashkeel_web_ui.py
```

Run the punctuation-only web UI:

```powershell
& $py debug_tools\punctuation_web_ui.py
```

Or use the launcher:

```powershell
.\debug_tools\launchers\run_punctuation_ui.ps1
```

Default local URLs:

```text
main tashkeel UI:     http://127.0.0.1:8765/
punctuation-only UI:  http://127.0.0.1:8776/
```

## Workspace Map

Keep these folders:

- `debug_tools/` - local UIs and launchers.
- `ML_Tashkeel/` - punctuation restoration training, eval, reports, and models.
- `scripts/` - dataset prep, CATT training, and eval helpers.
- `datasets/` - source and versioned split data.
- `external/` - CATT runtime/training checkouts and v4 exported checkpoint.
- `vendor/` - flat snapshots needed for backup context.
- `docs/` - older reports and catchup notes.

Ignored/generated areas include packaged apps, raw data drops, CATT-format build
outputs, runtime logs, Python caches, local model backups, and rollout session
transcripts.

## Notes

- Large model artifacts are stored with Git LFS.
- Portable packaged app folders are intentionally excluded from Git.
- Do not promote a punctuation checkpoint only from manual impressions; compare
  official eval and `punc_tests.txt` first.
