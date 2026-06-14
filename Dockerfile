# ==========================================
# STAGE 1: Build & Dependency Compilation
# ==========================================
FROM python:3.14-slim AS builder

WORKDIR /build

# Install system utilities required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into a localized wheelhouse
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# STAGE 2: Final Ephemeral Runtime Environment
# ==========================================
FROM python:3.14-slim AS runtime

WORKDIR /app

# Install git since our diff processing layer relies on git history
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application orchestration scripts and configurations
COPY config.json .
COPY src/ ./src/
COPY semgrep_rules/ ./semgrep_rules/

# Configure default container environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python"]
