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

# Use echo + python -c with escaped quotes — shell-safe multi-line command
RUN echo "from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='jayansh21/codesheriff-bug-classifier', \
    local_dir='/app/model/codesheriff', \
    local_dir_use_symlinks=False, \
    revision='main' \
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

# Copy the pre-downloaded model from model_downloader stage
COPY --from=model_downloader /app/model /app/model

# Copy application code and config
COPY config.json .
COPY src/ ./src/
COPY semgrep_rules/ ./semgrep_rules/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set model path for ModelManager to use
ENV MODEL_PATH=/app/model/codesheriff

# No fixed ENTRYPOINT: this image runs both `semgrep ...` (isolated pipx venv)
# and `python /app/src/*.py ...` (app venv). Callers must specify the full
# command, e.g.:
#   docker run <image> semgrep --config=... /workspace
#   docker run <image> python /app/src/triage_orchestrator.py --workspace /workspace
