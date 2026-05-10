# Sukoon Arabic Tashkeel Runtime

Sukoon is a compact Arabic diacritization runtime packaged as a Docker web app and HTTP API. It adds tashkeel to undiacritized Arabic text while keeping the deployment surface intentionally small: the repository contains only the inference server, built web UI, runtime model code, and the active Sukoon checkpoint.

This handoff build is intended for a private project integration. It does not include training datasets, research logs, experiment reports, generation scripts, or rejected checkpoints.

## What Is Included

| Component | Purpose |
| --- | --- |
| `Dockerfile` | Builds a CPU-only Docker image for local inference. |
| `docker-compose.yml` | Starts the web UI on `http://127.0.0.1:8765/`. |
| `container_demo/server.py` | Lightweight Python HTTP server and API. |
| `container_demo/webui/dist/` | Pre-built browser UI. |
| `runtime_models/.../catt_punct_v4/*.py` | Minimal Sukoon/CATT runtime code. |
| `catt_eo_ft_polished_v2/catt_eo_ft_best.pt` | Active Sukoon polished-v2 checkpoint. |

## Quick Start

Install Docker Desktop, then run:

```bash
git clone https://github.com/Dwekatious/Tashkeel.git
cd Tashkeel
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765/
```

Stop the container:

```bash
docker compose down
```

## API Usage

Health check:

```bash
curl http://127.0.0.1:8765/healthz
```

Diacritize text:

```bash
curl -X POST http://127.0.0.1:8765/api/diacritize \
  -H "Content-Type: application/json" \
  -d '{"model":"eo","text":"كتب الطالب الدرس."}'
```

Example response:

```json
{
  "model": "eo",
  "output": "كَتَبَ الطَّالِبُ الدَّرْس.",
  "device": "cpu",
  "segments": 1
}
```

## Runtime Profile

| Property | Value |
| --- | --- |
| Runtime model | Sukoon v2 / polished-v2 EO checkpoint |
| Model family | CATT-style character-level Arabic diacritizer |
| Deployment | Docker, CPU inference |
| Default port | `127.0.0.1:8765` |
| Container port | `7860` |
| Active checkpoint size | About 73 MB |
| Docker image size | About 1.58 GB after build, mostly CPU PyTorch |
| App payload copied into image | About 72 MB |

## Benchmark Summary

The numbers below are from the filtered Sukoon evaluation pipeline used during development. Higher quality is better. The filtered scorer ignores known non-task noise such as hamza spelling variants, accepted tanween-alif placement differences, pause-form final-haraka suppression before strong punctuation, and wasla/definite-article variants.

| Metric | Quality | Error |
| --- | ---: | ---: |
| DER | 98.92% | 1.08% |
| WER | 96.54% | 3.46% |

Dataset scale for the full filtered evaluation:

| Item | Count |
| --- | ---: |
| Aligned rows | 16,506 |
| Evaluated characters | 495,602 |
| Evaluated words | 113,955 |
| Batch-eval throughput | About 9,281 words/sec |

Quality chart:

```text
DER  98.92% | #################################################.
WER  96.54% | ################################################..
```

Source-level stress points:

| Source | DER Quality | WER Quality | Notes |
| --- | ---: | ---: | --- |
| General fixtures | 99.29% | 97.00% | Clean, representative samples. |
| Wikipedia | 92.47% | 72.54% | Hardest observed source. |
| Shamela | 95.16% | 85.78% | Classical/book-style text is harder. |
| Easy Arabic News | 95.60% | 86.64% | News style with varied sentence forms. |
| Quran | 96.89% | 89.54% | Specialized orthography and style. |

Container smoke test on CPU:

| Input | Output | Runtime |
| --- | --- | ---: |
| `كتب الطالب الدرس.` | `كَتَبَ الطَّالِبُ الدَّرْس.` | About 15 ms |

Benchmark notes:

- Batch-eval throughput was measured in the development evaluation environment, not inside the final Docker image.
- Docker runtime performance depends on the host CPU and Docker Desktop settings.
- The included image is CPU-only for simple cross-platform use.

## Architecture

```text
Browser UI
   |
   v
container_demo/server.py
   |
   v
LocalCATTInference
   |
   v
Sukoon polished-v2 checkpoint
```

Request flow:

1. Text is sent from the UI or `/api/diacritize`.
2. The server normalizes runtime options and selects Sukoon.
3. Long text is split into manageable segments.
4. Sukoon predicts Arabic diacritics.
5. The server returns the final diacritized text and basic telemetry.

## Repository Layout

```text
.
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── container_demo/
│   ├── index.html
│   ├── server.py
│   └── webui/dist/
└── runtime_models/
    └── premium/external/catt_punct_v4/
        ├── *.py
        └── catt_eo_ft_polished_v2/catt_eo_ft_best.pt
```

## Mac Notes

On Apple Silicon Macs, building locally is preferred:

```bash
docker compose up --build
```

This lets Docker build the image for the Mac's native architecture instead of running an image built on another machine through emulation.

## License

This repository is proprietary and private. See `LICENSE`.

Only the authorized recipient granted access by Mohanad Dwekat may use, modify, and integrate this runtime. Redistribution, publishing, resale, sublicensing, or sharing with additional parties is not allowed without explicit written permission from Mohanad Dwekat.
