# Multi-stage Dockerfile for RAG Chatbot System
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Create final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/vector_store /app/data && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser modules/ /app/modules/
COPY --chown=appuser:appuser .env.example /app/.env.example

# Create directories
RUN mkdir -p /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "modules.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
