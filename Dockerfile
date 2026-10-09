# Production Dockerfile for AAVAIL Revenue Projection API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and models
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY tests/ ./tests/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV LOG_DIR=/app/logs

EXPOSE 8000

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
