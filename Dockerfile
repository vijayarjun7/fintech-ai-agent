# ── Base ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────
# curl  → healthcheck
# gcc   → sentence-transformers native build
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Environment ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── Dependencies (separate layer — cached unless requirements.txt changes) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# ── Non-root user (security) ───────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Ports ──────────────────────────────────────────────────────────────────
EXPOSE 7860
EXPOSE 8000

# ── Healthcheck ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Default command: FastAPI ───────────────────────────────────────────────
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
