# Multi-stage Dockerfile for RAG Application
# Optimized for smaller image size and faster builds

# ─────────────────────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip

# Copy only dependency files first (layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies from uv.lock if available, otherwise from pyproject.toml
RUN if [ -f uv.lock ]; then \
        pip install uv && uv pip install --system -r pyproject.toml; \
    else \
        pip install -e .; \
    fi

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY pyproject.toml ./

# Create directories for data persistence
RUN mkdir -p /app/docs /app/index && \
    touch /app/index/.gitkeep

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash raguser && \
    chown -R raguser:raguser /app
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Default command: run API server
CMD ["python", "-m", "app.api"]
