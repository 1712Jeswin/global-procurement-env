# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Port 7860 as required for Hugging Face Spaces
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
# Rule 4 Core Requirement: --workers 1 for thread safety
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
