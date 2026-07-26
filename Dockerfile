# ── Stage 1: Dependencies ──────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install only dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: Runtime ───────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# System deps for matplotlib & SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default: run API server (override CMD for Gradio)
EXPOSE 8000
CMD ["python", "run_api.py"]
