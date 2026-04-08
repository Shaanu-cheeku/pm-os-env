# Dockerfile
# ─────────────────────────────────────────────
# Builds the PM-OS environment into a Docker image.
#
# Requirements:
#   - Must build successfully
#   - Must run with < 8GB RAM (we use ~200MB — very lightweight)
#   - Must expose a working environment on port 7860 (HuggingFace standard)
# ─────────────────────────────────────────────

# Use a slim Python 3.11 base — small, fast, modern
FROM python:3.11-slim

# ── System setup ────────────────────────────────────────────────────
# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files (cleaner container)
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output buffering (see logs immediately)
ENV PYTHONUNBUFFERED=1

# ── Install dependencies ─────────────────────────────────────────────
# Copy requirements first (Docker caches this layer if unchanged)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ───────────────────────────────────────────────
# Copy everything into /app inside the container
COPY . .

# ── Expose port ──────────────────────────────────────────────────────
# HuggingFace Spaces expects port 7860
EXPOSE 7860

# ── Start the FastAPI server ─────────────────────────────────────────
# This runs app.py which exposes /reset, /step, /state endpoints
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
