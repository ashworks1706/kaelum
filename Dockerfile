FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LLM_BASE_URL=http://vllm:8000/v1
ENV LLM_MODEL=qwen2.5:3b
ENV LLM_TEMPERATURE=0.7
ENV LLM_MAX_TOKENS=2048
ENV MAX_REFLECTION_ITERATIONS=2
ENV USE_SYMBOLIC_VERIFICATION=true
ENV DEBUG_VERIFICATION=false

RUN mkdir -p .kaelum/analytics .kaelum/active_learning .kaelum/routing .kaelum/calibration

EXPOSE 8080

CMD ["python", "-u", "run.py"]
