# Production Dockerfile for CPU-only Recommender Service
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ /app/src/
COPY configs/ /app/configs/
RUN pip install --no-cache-dir -e .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Uvicorn server
CMD ["uvicorn", "recsys.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
