# ═══════════════════════════════════════════════════════════════════
# Banking Agentic AI Framework — Lean Dockerfile
# Multistage build: ~280MB final image, ~60s cold start
# ═══════════════════════════════════════════════════════════════════

# Stage 0: React front end (client/react-app → build/)
FROM node:20-alpine AS ui
WORKDIR /ui
COPY client/react-app/package*.json ./
RUN npm ci --no-audit --no-fund
COPY client/react-app/ ./
RUN npm run build

# Stage 1: Dependencies (cached layer)
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (minimal)
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source, then the compiled UI
COPY . .
COPY --from=ui /ui/build /app/client/react-app/build

# Make installed packages available
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=8s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.gateway:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
