FROM python:3.11-slim

LABEL maintainer="Manny <manuelmello.dev@gmail.com>"
LABEL description="SeraphynAI - Quantum-Inspired Multi-Identity Consciousness System"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 seraphyn && \
    chown -R seraphyn:seraphyn /app

USER seraphyn

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "seraphynai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
