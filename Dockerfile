FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TASHKEEL_HOST=0.0.0.0 \
    TASHKEEL_PORT=7860 \
    TASHKEEL_ENABLE_PREMIUM=0 \
    TASHKEEL_DEFAULT_MODEL=eo \
    TASHKEEL_EO_LABEL="Sukoon" \
    TASHKEEL_EO_CHECKPOINT=/app/runtime_models/premium/external/catt_punct_v4/catt_eo_ft_polished_v2/catt_eo_ft_best.pt \
    TASHKEEL_MAX_CHARS=0 \
    TASHKEEL_EO_BATCH_SIZE=32 \
    TASHKEEL_EO_SPLIT_TARGET_CHARS=320 \
    TASHKEEL_EO_SPLIT_HARD_CHARS=520 \
    TASHKEEL_EO_SPLIT_SOFT_TARGET_WORDS=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.6.0

# Web UI (pre-built) + inference server only.
COPY container_demo/server.py /app/container_demo/server.py
COPY container_demo/index.html /app/container_demo/index.html
COPY container_demo/webui/dist/ /app/container_demo/webui/dist/

# Sukoon (EO) runtime only: no training scripts, logs, datasets, or rejected checkpoints.
COPY runtime_models/premium/external/catt_punct_v4/bw2ar.py /app/runtime_models/premium/external/catt_punct_v4/bw2ar.py
COPY runtime_models/premium/external/catt_punct_v4/checkpoint_compat.py /app/runtime_models/premium/external/catt_punct_v4/checkpoint_compat.py
COPY runtime_models/premium/external/catt_punct_v4/eo.py /app/runtime_models/premium/external/catt_punct_v4/eo.py
COPY runtime_models/premium/external/catt_punct_v4/eo_runtime.py /app/runtime_models/premium/external/catt_punct_v4/eo_runtime.py
COPY runtime_models/premium/external/catt_punct_v4/hybrid_splitter.py /app/runtime_models/premium/external/catt_punct_v4/hybrid_splitter.py
COPY runtime_models/premium/external/catt_punct_v4/local_inference.py /app/runtime_models/premium/external/catt_punct_v4/local_inference.py
COPY runtime_models/premium/external/catt_punct_v4/tashkeel_tokenizer.py /app/runtime_models/premium/external/catt_punct_v4/tashkeel_tokenizer.py
COPY runtime_models/premium/external/catt_punct_v4/transformer.py /app/runtime_models/premium/external/catt_punct_v4/transformer.py
COPY runtime_models/premium/external/catt_punct_v4/utils.py /app/runtime_models/premium/external/catt_punct_v4/utils.py
COPY runtime_models/premium/external/catt_punct_v4/catt_eo_ft_polished_v2/catt_eo_ft_best.pt /app/runtime_models/premium/external/catt_punct_v4/catt_eo_ft_polished_v2/catt_eo_ft_best.pt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3).read()"

CMD ["python", "container_demo/server.py", "--preload", "eo"]
