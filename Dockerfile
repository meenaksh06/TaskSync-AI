# Use a more robust base image to avoid apt-get issues in some environments
FROM python:3.12-slim

WORKDIR /app

# Install only essential build tools and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root
COPY requirements.txt .

# Install dependencies  (will include the spacy model now)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the actual backend code
COPY backend/ .

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app_enhanced:app", "--host", "0.0.0.0", "--port", "8000"]
