#!/bin/bash
set -e

echo "=========================================="
echo "Kaelum AI - Docker Setup"
echo "=========================================="

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"

echo ""
echo "Configuration:"
echo "  vLLM Model: $MODEL"
echo "  vLLM Port: 8000"
echo "  Kaelum Port: 8080"
echo ""

export VLLM_MODEL="$MODEL"
export KAELUM_MODEL="$MODEL"

echo "Building Kaelum container..."
docker-compose build kaelum

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM is ready!"
        break
    fi
    echo -n "."
    sleep 2
    if [ $i -eq 60 ]; then
        echo ""
        echo "Error: vLLM failed to start within 2 minutes"
        echo "Check logs with: docker-compose logs vllm"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "✓ Kaelum is running!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - vLLM API: http://localhost:8000"
echo "  - Kaelum App: http://localhost:8080"
echo ""
echo "Useful commands:"
echo "  View logs:       docker-compose logs -f kaelum"
echo "  Stop services:   docker-compose down"
echo "  Restart:         docker-compose restart"
echo "  Shell access:    docker-compose exec kaelum bash"
echo ""
echo "Test the system:"
echo '  docker-compose exec kaelum python -c "from kaelum import enhance; print(enhance(\"What is 2+2?\"))"'
echo ""
