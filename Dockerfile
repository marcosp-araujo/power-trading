# --- Stage 1: The Builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Prevent uv from creating a cache inside the image
ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/tmp/uv_cache 

WORKDIR /app

# Bind mounts keep the pyproject/lock files out of the final layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv_cache \
    uv sync --frozen --no-install-project --no-dev

# --- Stage 2: The Runtime (The only one that matters) ---
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Copy ONLY the site-packages and binaries, nothing else
COPY --from=builder /app/.venv /app/.venv

# Copy app code (respecting .dockerignore)
COPY models ./models
COPY src ./src
COPY data/time_series_15min.parquet ./data/time_series_15min.parquet
COPY /app.py ./

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]