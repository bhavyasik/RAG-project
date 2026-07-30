# ─────────────────────────────────────────────────────────────────────────────
# RAG Application — Optimised Multi-Stage Dockerfile
#
# Key optimisations:
#   1. Official uv binary   — fastest Python package installer (10-100x vs pip)
#   2. CPU-only PyTorch     — avoids downloading 3 GB+ of CUDA/NVIDIA packages
#   3. BuildKit cache mount — cached wheels survive image rebuilds (seconds, not
#                             minutes, after the very first build)
#   4. No build-essential   — all deps ship as pre-built wheels; no compilation
#   5. Multi-stage build    — builder layer discarded; runtime image stays slim
# ─────────────────────────────────────────────────────────────────────────────

# syntax=docker/dockerfile:1.4
# ─────────────────────────────────────────────────────────────
# Stage 1: Builder  — install all Python dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Pull the uv binary straight from the official image (no pip install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# Copy only the dependency manifest — changes here invalidate the install layer,
# but NOT the cache-mounted wheel store, so re-installs are still fast.
COPY pyproject.toml ./

# ── Dependency installation ───────────────────────────────────────────────────
# --mount=type=cache  : persists the uv wheel cache between builds on the host
# --extra-index-url   : fetches torch from the CPU-only PyTorch index (~200 MB)
#                       instead of the default PyPI CUDA build (~3 GB+)
# UV_LINK_MODE=copy   : use file copies instead of hardlinks (safer in Docker)
# --no-dev            : skip dev/test dependencies for a smaller image
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    uv pip install \
        --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        "torch>=2.0,<3" \
        -r pyproject.toml

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime  — tiny image with only what is needed to run
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgomp1 is the only native runtime lib required by sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed site-packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/ ./app/
COPY pyproject.toml ./

# Runtime directories (mounted as volumes in production)
RUN mkdir -p /app/docs /app/index

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Tell sentence-transformers/HuggingFace to cache models here
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash raguser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R raguser:raguser /app
USER raguser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

EXPOSE 8000
CMD ["python", "-m", "app.api"]
