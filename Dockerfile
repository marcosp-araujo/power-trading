# --- Stage 1: The Builder ---
FROM python:3.12-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest installer, install it and then remove it
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

# Set up the UV environment path correctly
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY ./pyproject.toml .

RUN uv sync --no-dev

# --- Stage 2: The Runtime (The only one that matters) ---
FROM python:3.12-slim-bookworm AS production

WORKDIR /app

# Copy ONLY the site-packages and binaries, nothing else
COPY --from=builder /app/.venv /app/.venv

# Copy app code (respecting .dockerignore)
COPY models ./models
COPY src ./src
COPY /app.py ./


ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]