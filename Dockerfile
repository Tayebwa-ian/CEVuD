# ==========================================
# STAGE 1: Build & Dependency Compilation
# ==========================================
FROM python:3.14-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Leverage BuildKit cache mount for pip downloads
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# ==========================================
# STAGE 2: Download and Cache SLM Model (Critical Optimization)
# ==========================================
FROM python:3.14-slim AS model_downloader

# Install required tools for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install transformers and huggingface_hub for model download
RUN pip install --no-cache-dir transformers huggingface_hub

# Create model directory
RUN mkdir -p /app/model

# Download BOTH models into the HuggingFace *cache* layout under
# /app/model_cache. ModelManager resolves weights via `cache_dir`
# (config -> paths.model_cache_dir), which we repoint at
# /app/model_cache below — so the baked weights are actually
# reused at runtime (no per-run HuggingFace download) and the
# pipeline runs fully offline. snapshot_download's `local_dir`
# must be the exact `models--<org>--<repo>` path so that
# `from_pretrained(cache_dir='/app/model_cache')` finds it.
RUN echo "from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='jayansh21/codesheriff-bug-classifier', \
    local_dir='/app/model_cache/models--jayansh21--codesheriff-bug-classifier', \
    local_dir_use_symlinks=False, \
    revision='main' \
); \
snapshot_download( \
    repo_id='microsoft/codebert-base', \
    local_dir='/app/model_cache/models--microsoft--codebert-base', \
    local_dir_use_symlinks=False \
    )" | python3
# ==========================================
# STAGE 3: Final Ephemeral Runtime Environment
# ==========================================
FROM python:3.14-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    pipx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install semgrep into its own isolated virtualenv via pipx.
# semgrep pins wcmatch~=8.3, which directly conflicts with deepagents'
# wcmatch>=10.1 requirement — they cannot share one environment.
# pipx keeps semgrep's dependency tree fully separate; only the
# `semgrep` executable is exposed on PATH (into /root/.local/bin,
# alongside the --user-installed app scripts, with no site-packages
# collision since pipx's venv lives under /root/.local/pipx/venvs/).
ENV PIPX_HOME=/root/.local/pipx
ENV PIPX_BIN_DIR=/root/.local/bin
RUN pipx install semgrep==1.166.0

# Copy the pre-downloaded model cache from model_downloader stage
COPY --from=model_downloader /app/model_cache /app/model_cache

# Copy application code and config
COPY config.json .
COPY src/ ./src/
COPY semgrep_rules/ ./semgrep_rules/

# Optional: copy a custom-trained model into the image.
# Pass --build-arg CUSTOM_MODEL_PATH=/path/to/training_output/latest/model
# at build time to bake your fine-tuned weights into the image.
ARG CUSTOM_MODEL_PATH=""
RUN if [ -n "$CUSTOM_MODEL_PATH" ]; then \
        echo "[*] Baking custom model from $CUSTOM_MODEL_PATH"; \
        mkdir -p /app/custom_model; \
        cp -r "$CUSTOM_MODEL_PATH"/* /app/custom_model/; \
    fi

# Point ModelManager's cache_dir at the baked model cache so the
# pre-downloaded weights are reused (no network, fast re-runs).
# If a custom model was baked, repoint the classifier to it.
RUN python -c "import json; p='config.json'; d=json.load(open(p)); \
d['paths']['model_cache_dir']='/app/model_cache'; \
import os; \
custom='/app/custom_model'; \
if os.path.exists(custom) and os.listdir(custom): \
    d['models']['classifier_model']=custom; \
    print('[+] Custom classifier model detected at', custom); \
json.dump(d, open(p,'w'), indent=2)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run fully offline against the baked model cache.
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# No fixed ENTRYPOINT: this image runs both `semgrep ...` (isolated pipx venv)
# and `python /app/src/*.py ...` (app venv). Callers must specify the full
# command, e.g.:
#   docker run <image> semgrep --config=... /workspace
#   docker run <image> python /app/src/triage_orchestrator.py --workspace /workspace
