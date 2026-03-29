FROM python:3.12-slim

WORKDIR /app

# Enable a swap file inside the container logic (simulated by memory flags)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies - Use a single thread to save RAM during build
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models in build stage
COPY backend/download_models.py .
RUN python download_models.py && rm download_models.py

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the actual backend code
COPY backend/ .

EXPOSE 8000

# High-performance, low-memory production server
# --workers 1 is CRITICAL for 512MB RAM
CMD ["uvicorn", "app_enhanced:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "5"]
