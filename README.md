# Kaelum

A production-ready AI reasoning framework with human-in-the-loop feedback, neural routing, Monte Carlo Tree Search, domain-specific verification, and continuous learning for robust multi-step problem solving.

<img width="1983" height="1098" alt="image" src="https://github.com/user-attachments/assets/97f5601e-e660-44b1-9338-80308e0d80d4" />
<img width="1983" height="915" alt="image" src="https://github.com/user-attachments/assets/1d810ebb-496f-494b-9f4a-cb3022dd22fe" />
<img width="1983" height="844" alt="image" src="https://github.com/user-attachments/assets/6b000d29-d8bc-4219-8157-de5bf966f229" />

**What is this?** Kaelum is an AI reasoning system that combines multiple AI techniques to solve complex problems step-by-step. It learns from human feedback to continuously improve, uses neural routing to select expert workers, and verifies answers before returning them.

**Core Pipeline:**
- Query → Cache Lookup → Feedback-Enhanced Neural Router → Expert Worker (LATS) → Verification → Result
- Six specialized workers: Math, Logic, Code, Factual, Creative, Analysis
- Human-in-the-Loop feedback for continuous improvement
- Quality-aware semantic cache for instant retrieval

---

## Key Features

- **Human Feedback Integration**: Rate worker selection, answer quality, and reasoning steps. Feedback directly adjusts worker probabilities and improves future performance.
- **Neural Router**: Learns from outcomes to select optimal workers for each query type.
- **Intelligent Caching**: Semantic similarity search with quality filtering and LLM validation.
- **LATS Search**: Monte Carlo Tree Search explores multiple solution paths with early pruning.
- **Multi-Layer Verification**: Domain-specific verification ensures correctness.
- **Interactive Visualizations**: Tree visualizations, live metrics, router analytics, and feedback dashboards.

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
```

### 2. Install Dependencies

```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies  
cd frontend
npm install
cd ..
```

### 3. Start vLLM Backend (Recommended)

```bash
# Install vLLM
pip install vllm

# Start server with a balanced model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7
```

### 4. Start Kaelum Web Interface

**Option 1: Automatic (recommended)**

```bash
./start_demo.sh
```

**Option 2: Manual**

```bash
# Terminal 1 - Start backend (port 5000)
cd backend
python app.py

# Terminal 2 - Start frontend (port 3000)
cd frontend
npm run dev
```

Then open http://localhost:3000 in your browser.

---

## How It Works

### Pipeline Overview

```
Query Input
    ↓
[1] Query Embedding & Cache Lookup
    └─ Cache hit? → Return result (0.001s)
    ↓
[2] Neural Router
    └─ Select: Worker + Tree Depth + Simulations
    ↓
[3] LATS Search (Monte Carlo Tree Search)
    └─ Explore solution paths with UCT selection
    ↓
[4] Verification
    └─ Validate solution correctness
    ↓
[5] Cache Result & Update Router
    └─ Store for future use
```

### Components

1. **Quality-Aware Semantic Cache**: Finds similar queries (cosine ≥0.85), validates with LLM, serves verified results instantly.

2. **Neural Router**: 398 → 256 → 128 layer network that selects optimal worker based on query features. Learns from outcomes and human feedback.

3. **Six Expert Workers**: Math, Logic, Code, Factual, Creative, Analysis - each specialized for specific task types.

4. **LATS Search**: Monte Carlo Tree Search with UCT selection explores multiple reasoning paths, prunes low-quality branches.

5. **Verification**: Domain-specific verification (symbolic for math, execution for code, factual checking) ensures correctness.

6. **Human Feedback Loop**: Collects feedback on worker selection, answer quality, and reasoning steps to continuously improve the system.

---

## Supported LLMs

Kaelum is **model-agnostic** and works with any OpenAI-compatible API.

### Recommended Models

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| SmolLM2 | 1.7B | 3 GB | Fast | Edge/Mobile, testing |
| Qwen2.5 | 7B | 14 GB | Balanced | General reasoning |
| Qwen2.5 | 14B | 28 GB | Strong | Complex reasoning |
| DeepSeek-R1 | 7B | 14 GB | Best | Advanced reasoning |

### vLLM Setup

```bash
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

### Cloud APIs

```bash
# Set environment variable
export OPENAI_API_KEY=your-key-here

# Update config in web UI or via API
curl -X POST http://localhost:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"base_url": "https://api.openai.com/v1", "model": "gpt-4"}'
```

---

## Project Structure

```
kaelum/
├── backend/
│   └── app.py              # Flask REST API
├── frontend/
│   ├── app/components/     # React components
│   └── package.json
├── core/
│   ├── reasoning.py        # LLM client
│   ├── config.py           # Configuration
│   ├── detectors/          # Query classifiers
│   ├── cache/              # Semantic cache
│   ├── verification/       # Multi-layer verification
│   └── workers/            # Expert workers
├── runtime/
│   └── orchestrator.py     # Main orchestration
├── kaelum.py               # Python API
└── .kaelum/                # Persistent data
    ├── routing/            # Router training data
    ├── cache/              # Cached LATS trees
    └── analytics/          # Performance metrics
```

---

## Configuration

Configuration is managed through the web interface or via API.

**Update via API:**

```bash
curl -X POST http://localhost:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"temperature": 0.8, "max_tree_depth": 8}'
```

**Key Settings:**
- `enable_routing`: Use neural router (default: true)
- `use_symbolic_verification`: Enable math verification (default: true)
- `max_reflection_iterations`: Self-correction attempts (default: 2)
- `router_exploration_rate`: Exploration vs exploitation (default: 0.1)

---

## Python API Example

```python
from kaelum import kaelum_enhance_reasoning, set_reasoning_model

# Configure the system
set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    enable_routing=True
)

# Process a query
result = kaelum_enhance_reasoning("What is the derivative of x² + 3x?")

print(f"Answer: {result['suggested_approach']}")
print(f"Worker: {result['worker_used']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## Research & References

- [Browne et al. (2012): "A Survey of Monte Carlo Tree Search Methods"](https://ieeexplore.ieee.org/document/6145622)
- [Wei et al. (2022): "Chain-of-Thought Prompting"](https://arxiv.org/abs/2201.11903)
- [Yao et al. (2023): "Tree of Thoughts"](https://arxiv.org/abs/2305.10601)
- [Shinn et al. (2023): "Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366)
- [Reimers & Gurevych (2019): "Sentence-BERT"](https://arxiv.org/abs/1908.10084)
