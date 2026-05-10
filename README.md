# Sukoon

<p align="center">
  <img src="assets/sukoon-logo.png" alt="Sukoon logo" width="220">
</p>

<p align="center">
  <img src="assets/sukoon-banner.png" alt="Sukoon Arabic tashkeel model banner" width="100%">
</p>

<h3 align="center">A specialist Arabic tashkeel model engineered for accuracy, efficiency, and speech pipelines.</h3>

<p align="center">
  <b>up to 99.29% DER quality</b> · <b>up to 97.00% WER quality</b> · <b>9,281 words/sec GPU</b> · <b>438 words/sec CPU</b> · <b>~73 MB checkpoint</b>
</p>

Sukoon is a purpose-built Arabic diacritization model. It takes undiacritized Arabic text and returns the same text with tashkeel, using a compact encoder-style character system instead of a giant general-purpose text generator.

The thesis is simple: **Arabic tashkeel should not require a frontier LLM.** For this task, a dedicated model can be sharper, cheaper, faster, easier to control, and dramatically easier to integrate into real products.

## Executive Snapshot

<p align="center">
  <img src="assets/sukoon-scorecard.svg" alt="Sukoon scorecard" width="100%">
</p>

| Dimension | Sukoon Position |
| --- | --- |
| Accuracy | **Up to 99.29% DER quality** and **97.00% WER quality** under task-aware scoring. |
| Efficiency | A **~73 MB** active checkpoint, over **350x smaller** than a 26 GB local LLM-class checkpoint. |
| Throughput | **Up to 9,281 words/sec** on RTX 6000 Ada GPU and **438 words/sec** on a Ryzen 5 5600G CPU service-style workload. |
| Cost profile | Pricing is **TBD**, designed for a dramatically cheaper specialist tashkeel tier than frontier LLMs. |
| Integration | Clean text-in/text-out behavior, ideal for preprocessing, publishing, education, and TTS. |
| Hardware | Comfortable on ordinary laptops, small servers, and low-cost CPU deployments. |
| Stability | No prompt dance, no creative rewriting, no chatbot behavior. |

## Accuracy

Sukoon is benchmarked with task-aware tashkeel scoring, so the headline numbers focus on the marks the model is actually asked to predict. On clean, punctuation-correct text, this produces a very strong production profile:

<p align="center">
  <img src="assets/accuracy-bars.svg" alt="Sukoon DER and WER quality chart" width="88%">
</p>

| Metric | Quality |
| --- | ---: |
| DER | **up to 99.29%** |
| WER | **up to 97.00%** |

<details>
<summary>Scoring notes</summary>

The advertised DER/WER scoring ignores hamza spelling because Sukoon is not asked to correct hamza. It also ignores pause-form final-haraka differences immediately before punctuation, because the runtime rule removes the final short vowel before strong punctuation. Tanween is counted by whether the correct tanween is present, not by strict attachment position. These results assume the input text already has correct punctuation; if punctuation is missing or wrong, results may vary.

</details>

Full filtered evaluation scale:

| Evaluation Item | Count |
| --- | ---: |
| Aligned rows | 16,506 |
| Evaluated characters | 495,602 |
| Evaluated words | 113,955 |

## Hardware Performance

| Runtime Mode | Hardware | Throughput |
| --- | --- | ---: |
| GPU batch eval | NVIDIA RTX 6000 Ada Generation, batch size 64 | **up to 9,281 words/sec** |
| CPU service-style local run | AMD Ryzen 5 5600G, batch size 32 | **up to 438.5 words/sec** |

That gives Sukoon a practical deployment spread: tiny CPU services for low-cost integration, and GPU batching that scales to high-volume preprocessing. The GPU path measured here is about **21x** the local CPU throughput, while still using the same compact checkpoint family.

## Pricing Position

Sukoon pricing is **TBD**. The commercial target is clear: tashkeel should be priced like a specialized utility, not like frontier reasoning.

<p align="center">
  <img src="assets/pricing-comparison.svg" alt="API pricing comparison chart" width="88%">
</p>

Approximate full-pass cost for **1M words processed**:

| Model/API | Approx / 1M words | Estimate Basis | Notes |
| --- | ---: | --- | --- |
| **Sukoon** | TBD | Specialist tashkeel API | Intended to sit in a much cheaper specialist tier. |
| GPT-5.5 | **≈ $46.55** | $6.65 input + $39.90 output | OpenAI standard API pricing. |
| Claude Opus 4.7 | **≈ $39.90** | $6.65 input + $33.25 output | Anthropic standard Claude API pricing. |

Public reference prices checked on **2026-05-10**. Approximation assumes **1M words ≈ 1.33M input tokens + 1.33M output tokens** for a full text-in/text-out tashkeel pass. Real tokenizer billing varies by text style, punctuation, and diacritics.

Sources: [OpenAI API pricing](https://openai.com/api/pricing/) and [Anthropic Claude API pricing](https://platform.claude.com/docs/en/about-claude/pricing).

The point of Sukoon is not to compete with LLMs at everything. It is to make one high-volume Arabic operation inexpensive enough to use everywhere.

## Why Not Just Use A SOTA LLM?

| Requirement | Sukoon | SOTA LLM API | Local 30B+ LLM |
| --- | --- | --- | --- |
| Arabic tashkeel specialization | **Native purpose** | Prompted behavior | Prompted behavior |
| Text preservation | **Designed to preserve text** | Can rewrite or mutate text | Can rewrite or mutate text |
| Cost at scale | **Specialist pricing TBD** | High token economics | Hardware-heavy |
| Hardware footprint | **Tiny checkpoint** | Remote dependency | Tens of GB typical |
| Latency | **Small deterministic pipeline** | Network + generation latency | Heavy autoregressive decoding |
| TTS pipeline fit | **Excellent preprocessor** | Expensive and prompt-sensitive | Slow and memory-heavy |
| Operational risk | **Narrow output contract** | Broad model behavior | Broad model behavior |

General local LLMs can produce impressive Arabic, but tashkeel is not a general generation problem. The model must preserve every base character, avoid paraphrasing, handle punctuation, and output dense diacritics consistently. Sukoon is built for that pressure point directly.

<p align="center">
  <img src="assets/production-fit-map.svg" alt="Arabic tashkeel production fit map" width="88%">
</p>

## Runs On A Potato

Sukoon is intentionally small. The active checkpoint is about **73 MB**, and the runtime is CPU-first. That makes it suitable for:

| Environment | Fit |
| --- | --- |
| Developer laptop | Easy local testing. |
| Small VPS | Practical low-cost deployment. |
| Internal service | Private text processing. |
| Edge-ish CPU box | Possible where large LLMs are unrealistic. |
| TTS preprocessing pipeline | Excellent fit before acoustic/prosody stages. |

For TTS, Sukoon is especially valuable because Arabic speech quality depends heavily on vowels, endings, shadda, sukoon, tanween, and pause behavior. A cheap tashkeel preprocessor can improve downstream pronunciation without dragging a giant LLM into every synthesis request.

The important operational detail: Sukoon is measured in **words per second**, not slow autoregressive token-by-token generation. It is a compact encoder pass built for text preparation, so it can sit directly before a TTS engine without becoming the bottleneck.

## Architecture

```mermaid
flowchart LR
    A[Undiacritized Arabic] --> B[Unicode cleaning and normalization]
    B --> C[Arabic-aware segmentation]
    C --> D[Character encoder]
    D --> E[Contextual transformer layers]
    E --> F[Diacritic classifier]
    F --> G[Orthographic post-processing]
    G --> H[Diacritized Arabic]

    classDef input fill:#1c1917,stroke:#fb923c,stroke-width:2px,color:#fff7ed;
    classDef core fill:#111827,stroke:#ef4444,stroke-width:2px,color:#fee2e2;
    classDef output fill:#2a0f0f,stroke:#f97316,stroke-width:2px,color:#ffedd5;
    class A,H input;
    class B,C,G output;
    class D,E,F core;
```

Sukoon is a specialized character-level encoder for Arabic diacritization. It does not generate a sentence from scratch. It studies the input stream at the character level, builds contextual representations, and selects the correct diacritic marks for each Arabic base character.

That design matters. Arabic tashkeel is dense, local, and grammatical. A single internal haraka can change form or meaning; a final short vowel can alter case; shadda changes the consonantal structure; tanween and sukoon affect pronunciation and downstream speech synthesis. Sukoon treats these as first-class prediction targets, not decoration.

The deployed model is deliberately narrow and disciplined: it predicts the missing vocalization layer while preserving the base text. The system work around it is where much of the lift happened: Unicode cleanup, punctuation-aware segmentation, pause-form handling, filtered evaluation, source-level weakness analysis, and repeated checkpoint selection until accuracy, speed, and deployability lined up.

### What We Changed And Hardened

| Area | Work Done |
| --- | --- |
| Character vocabulary | Preserved Arabic base characters while expanding the diacritic decision surface. |
| Sequence handling | Added punctuation-aware segmentation so longer passages stay stable. |
| Pause behavior | Suppressed final short vowels before strong punctuation where pause-form output is preferred. |
| Evaluation | Built filtered DER/WER scoring to separate real tashkeel errors from spelling/orthographic noise. |
| Training data | Iterated through polished corpora, hard-source analysis, and weakpoint extraction. |
| Runtime surface | Reduced the deployed model to the minimum inference path: no training machinery in the handoff. |

This was not a random fine-tune. The hard work was making the system behave like a product: stable input preservation, meaningful filtered scoring, source-level weakness analysis, inference segmentation, checkpoint selection, and a tiny deployable runtime that can be dropped into real Arabic speech and publishing workflows.

## Training Process

```mermaid
flowchart TD
    A[Raw Arabic corpora] --> B[Cleaning and normalization]
    B --> C[Diacritized training pairs]
    C --> D[Balanced character-level batches]
    D --> E[Encoder training]
    E --> F[Filtered DER/WER evaluation]
    F --> G[Critical error analysis]
    G --> H[Data polishing]
    H --> D
    F --> I[Sukoon checkpoint selection]
    I --> J[Compact production runtime]
```

Training was guided by repeated measurement, not vibes. Each cycle exposed where the model was weak, separated harmless orthographic mismatch from real tashkeel errors, and pushed the checkpoint toward the highest practical blend of accuracy, speed, and deployability.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Builds the CPU runtime image. |
| `docker-compose.yml` | Starts the local service for testing. |
| `container_demo/server.py` | Python web server and inference entrypoint. |
| `container_demo/webui/dist/` | Pre-built browser UI assets. |
| `runtime_models/premium/external/catt_punct_v4/` | Minimal Sukoon runtime files. |
| `runtime_models/.../catt_eo_ft_best.pt` | Active model checkpoint. |

## Run

Install Docker Desktop, then:

```bash
git clone https://github.com/Dwekatious/Tashkeel.git
cd Tashkeel
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765/
```

Stop:

```bash
docker compose down
```
