# Dockerfile content with recent updates

# Base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONPATH /app

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Health check configuration
HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1

# Start the application
CMD ["python", "server/quantum_api.py"]