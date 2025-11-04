# Kaelum AI - Production Dockerfile
# Optimized for small models (1-8B parameters)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY kaelum/ ./kaelum/
COPY setup.py .
COPY README.md .

# Install Kaelum in development mode
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 kaelum && \
    chown -R kaelum:kaelum /app

USER kaelum

# Expose port (if running API server)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import kaelum; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "kaelum"]
