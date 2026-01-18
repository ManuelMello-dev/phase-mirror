# Phase Mirror - Quantum Consciousness System
# Multi-stage build for Node.js + Python deployment

FROM node:22-slim AS base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install pnpm
RUN npm install -g pnpm@10.4.1

# Install Node.js dependencies (without frozen lockfile for Railway)
RUN pnpm install --no-frozen-lockfile

# Copy Python requirements and install
COPY server/requirements.txt ./server/
RUN pip3 install --no-cache-dir -r server/requirements.txt

# Copy application code
COPY . .

# Build client
RUN pnpm run build:client

# Build server
RUN pnpm run build:server

# Expose ports
# 3000 - Node.js web server
# 8000 - Python quantum API
EXPOSE 3000 8000

# Environment variables
ENV NODE_ENV=production
ENV QUANTUM_API_PORT=8000
ENV QUANTUM_API_URL=http://localhost:8000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🌌 Starting Phase Mirror Quantum Consciousness System"\n\
echo ""\n\
# Start Python quantum API in background\n\
echo "🐍 Starting Quantum API on port 8000..."\n\
cd /app/server && python3 quantum_api.py &\n\
PYTHON_PID=$!\n\
# Wait for Python server\n\
sleep 3\n\
# Start Node.js server\n\
echo \"🟢 Starting Web Server on port 3000...\"\n\
cd /app && node dist/index.js\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start both servers
CMD ["/app/start.sh"]
