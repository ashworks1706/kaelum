# 🚀 Kaelum AI - Deployment Guide

Production-ready deployment guide for Kaelum v1.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Running in Production](#running-in-production)
6. [Monitoring & Logs](#monitoring--logs)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)

---

## ⚡ Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f kaelum
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -e .

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v0.3 \
    --port 8000 \
    --gpu-memory-utilization 0.7

# In another terminal, run Kaelum
python -c "
from kaelum import set_reasoning_model, enhance
set_reasoning_model(base_url='http://localhost:8000/v1')
print(enhance('Calculate 15% of $89.97'))
"
```

---

## 💻 System Requirements

### Minimum (Development)

- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: 6GB VRAM (RTX 3060, 4060)
- **Storage**: 10GB
- **OS**: Linux, macOS, Windows (WSL2)

### Recommended (Production)

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: 8-12GB VRAM (RTX 4070, A4000)
- **Storage**: 20GB SSD
- **OS**: Linux (Ubuntu 22.04+)

### Enterprise (High Volume)

- **CPU**: 16+ cores
- **RAM**: 32GB+
- **GPU**: 24GB+ VRAM (RTX 4090, A5000, A6000)
- **Storage**: 50GB NVMe SSD
- **OS**: Linux with NVIDIA drivers

---

## 🔧 Installation Methods

### Docker Installation (Recommended)

#### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

#### Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d
```

### Manual Installation

#### 1. Install Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Kaelum
pip install -e .

# Install vLLM (for local reasoning engine)
pip install vllm
```

#### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# macOS
brew install python@3.11

# Verify NVIDIA drivers (for GPU)
nvidia-smi
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Reasoning Engine
KAELUM_REASONING_URL=http://localhost:8000/v1
KAELUM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v0.3

# Performance
KAELUM_TEMPERATURE=0.7
KAELUM_MAX_TOKENS=2048
KAELUM_MAX_REFLECTION_ITERATIONS=2

# Verification
KAELUM_USE_SYMBOLIC_VERIFICATION=true
KAELUM_USE_FACTUAL_VERIFICATION=false
```

### Model Selection

Choose based on your GPU VRAM:

| Model | VRAM Required | Speed | Accuracy | Use Case |
|-------|--------------|-------|----------|----------|
| TinyLlama 1.1B | 2GB | ⚡⚡⚡ | ⭐⭐ | Testing, ultra-low latency |
| Qwen 1.5B | 3GB | ⚡⚡⚡ | ⭐⭐⭐ | **Recommended for production** |
| Phi-3 Mini | 4GB | ⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| Qwen 7B | 8GB | ⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |
| Mistral 7B | 8GB | ⚡ | ⭐⭐⭐⭐ | General reasoning |

### vLLM Server Configuration

```bash
# Basic (TinyLlama)
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v0.3 \
    --port 8000 \
    --gpu-memory-utilization 0.7

# Production (Qwen 1.5B with 4-bit quantization)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.85

# High accuracy (Qwen 7B)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
```

---

## 🏭 Running in Production

### Using systemd (Linux)

Create `/etc/systemd/system/kaelum-vllm.service`:

```ini
[Unit]
Description=Kaelum vLLM Reasoning Engine
After=network.target

[Service]
Type=simple
User=kaelum
WorkingDirectory=/opt/kaelum
Environment="PATH=/opt/kaelum/venv/bin"
ExecStart=/opt/kaelum/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kaelum-vllm
sudo systemctl start kaelum-vllm
sudo systemctl status kaelum-vllm
```

### Using Docker Compose (Production)

```bash
# Production compose file
docker-compose -f docker-compose.prod.yml up -d

# Auto-restart on failure
docker-compose -f docker-compose.prod.yml up -d --scale kaelum=2

# View logs
docker-compose logs -f --tail=100

# Monitor resource usage
docker stats
```

### Load Balancing (High Volume)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm-1
      - vllm-2

  vllm-1:
    image: vllm/vllm-openai:latest
    # ... config ...
    
  vllm-2:
    image: vllm/vllm-openai:latest
    # ... config ...
```

---

## 📊 Monitoring & Logs

### View Logs

```bash
# Docker logs
docker-compose logs -f kaelum
docker-compose logs -f vllm

# Manual deployment logs
tail -f /var/log/kaelum/kaelum.log
tail -f /var/log/kaelum/vllm.log
```

### Health Checks

```bash
# Check vLLM is running
curl http://localhost:8000/health

# Test reasoning endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Metrics

Monitor these key metrics:

- **Latency**: P50, P95, P99 response times
- **Throughput**: Queries per second
- **GPU Utilization**: Should be 70-90%
- **Memory Usage**: Watch for OOM errors
- **Error Rate**: Should be <1%

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.6  # Instead of 0.9

# Use smaller model
--model TinyLlama/TinyLlama-1.1B-Chat-v0.3  # Instead of 7B

# Enable quantization
--quantization awq  # 4-bit quantization
```

#### 2. Slow Response Times

```bash
# Increase batch size
--max-num-seqs 64  # Instead of 32

# Reduce max context length
--max-model-len 2048  # Instead of 4096

# Use faster model
--model Qwen/Qwen2.5-1.5B-Instruct  # Instead of 7B
```

#### 3. Connection Refused

```bash
# Check if vLLM is running
ps aux | grep vllm

# Check port availability
netstat -tulpn | grep 8000

# Test with curl
curl http://localhost:8000/health
```

#### 4. Model Download Fails

```bash
# Pre-download model
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v0.3')
"

# Use cached model
--model /path/to/cached/model
```

---

## 🔒 Security Best Practices

### 1. API Authentication

```python
# Add API key authentication
from kaelum import set_reasoning_model

set_reasoning_model(
    base_url="http://localhost:8000/v1",
    api_key="your-secure-api-key"
)
```

### 2. Network Security

```bash
# Restrict vLLM to localhost only
--host 127.0.0.1  # Not 0.0.0.0

# Use firewall
sudo ufw allow from 10.0.0.0/8 to any port 8000  # Internal network only
```

### 3. Resource Limits

```yaml
# docker-compose.yml
services:
  vllm:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
```

### 4. Audit Logging

```python
# Enable audit trail
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/kaelum/audit.log'),
        logging.StreamHandler()
    ]
)
```

---

## 📦 Data Requirements

### No External Data Required!

Kaelum v1 **does not require downloading external datasets**. Everything needed:

✅ **Models**: Auto-downloaded from Hugging Face on first run  
✅ **SymPy**: Bundled with Python (symbolic math verification)  
✅ **No training data**: Kaelum uses pre-trained models as-is

### Optional: Custom Prompts

You can customize reasoning prompts (no data download needed):

```python
from kaelum import set_reasoning_model

set_reasoning_model(
    reasoning_system_prompt="You are a precise reasoning assistant...",
    reasoning_user_template="Break down this problem: {query}"
)
```

---

## 🚦 Production Checklist

Before deploying to production:

- [ ] Environment variables configured (`.env` file)
- [ ] vLLM server tested and healthy
- [ ] GPU drivers installed and verified
- [ ] Logging configured and working
- [ ] Health checks passing
- [ ] Resource limits set (CPU, memory, GPU)
- [ ] Firewall rules configured
- [ ] Monitoring setup (optional but recommended)
- [ ] Backup strategy defined
- [ ] Documentation updated with your configs

---

## 📞 Support

- **GitHub Issues**: [github.com/ashworks1706/KaelumAI/issues](https://github.com/ashworks1706/KaelumAI/issues)
- **Documentation**: [github.com/ashworks1706/KaelumAI/docs](https://github.com/ashworks1706/KaelumAI/tree/main/docs)
- **Discord**: Coming soon

---

**Ready to ship! 🚀**
