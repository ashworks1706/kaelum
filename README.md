# Kaelum

A production-ready reasoning framework combining neural routing, Monte Carlo Tree Search, domain-specific verification, and self-reflection for robust multi-step problem solving.

**What is this?** Kaelum is an AI reasoning system that combines multiple AI techniques to solve complex problems step-by-step. It's like having multiple expert assistants (math, code, logic, etc.) working together, where each assistant explores different solution paths and the system verifies answers before returning them.

Core concepts:

- Query → **Cache Lookup (quality-filtered)** → Neural Router → Expert Worker (LATS with pruning) → Verification → Enhanced Router Feedback → Result
- Six specialized workers: Math, Logic, Code, Factual, Creative, Analysis
- **MCTS** (Monte Carlo Tree Search): A search algorithm that explores multiple solution paths by building a tree of possibilities, with early pruning of low-performing branches
- **Quality-aware semantic cache**: Stores previously solved high-quality problems using AI embeddings (numerical representations of meaning) for instant retrieval, checked BEFORE routing for maximum efficiency
- Continuous learning: router trains on enhanced feedback (avg rewards, depth, simulations); thresholds are F1-optimized

## 📚 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start) ⚡
- [Supported LLMs](#supported-llms) 🦙
  - [Recommended Models](#-recommended-models)
  - [vLLM Setup (Recommended)](#-vllm-setup-recommended)
  - [Cloud APIs](#️-cloud-apis-alternative)
  - [Other Deployment Options](#-other-deployment-options)
  - [Model Recommendations by Use Case](#-model-recommendations-by-use-case)
- [Detailed Setup Guide](#detailed-setup-guide) 📖
  - [vLLM + Kaelum](#step-by-step-vllm--kaelum)
  - [Multi-GPU Setup](#multi-gpu-setup-with-vllm)
  - [Quick Testing with Ollama](#alternative-quick-testing-with-ollama)
- [Configuration Options](#configuration-options)
- [Complete Workflow](#complete-workflow-from-query-to-answer)
- [Python API Examples](#example-python-api)
- [Troubleshooting](#troubleshooting-) 🔧
- [Research &amp; References](#research--references)

---

## Features

### 🧠 Core Intelligence

- **Neural Router with Enhanced Feedback**: Deep learning model (398→256→128 architecture) that learns from rich signals:
  - **Input features**: Query embeddings (384-dim) + structural features (length, math symbols, code keywords, etc.)
  - **Outputs**: Worker selection probabilities, optimal tree depth (3-10), simulation count (5-25)
  - **Learning signals**: Enhanced feedback with average tree rewards, actual depth/simulations used, success/failure
  - **Training**: Gradient descent after 32 outcomes, continually improves worker selection accuracy
  - **Effect**: System gets smarter with every query, learns domain patterns automatically

- **Six Specialized Expert Workers**: Each worker has domain-optimized prompting, scoring, and verification:
  - **Math Worker**: SymPy symbolic verification (derivatives, integrals, equations), rewards step-by-step algebraic reasoning
  - **Logic Worker**: Semantic coherence checks, premise-conclusion validation, rewards structured argumentation
  - **Code Worker**: AST parsing (Python/JS/TS), syntax validation, execution sandboxing, rewards clean documented code
  - **Factual Worker**: Information completeness scoring, joint embedding validation, rewards specific cited evidence
  - **Creative Worker**: Vocabulary diversity metrics, coherence detection, rewards originality + structure
  - **Analysis Worker**: Depth scoring, keyword presence, rewards comprehensive multi-perspective analysis

### 🌳 LATS - Language Agent Tree Search with Pruning

- **Monte Carlo Tree Search Adaptation**: Explores multiple reasoning paths before committing to an answer
  - **UCT Selection**: Balances exploitation (Q/N) vs exploration (c×√(ln N_parent / N_node)) with c=√2
  - **Early Pruning**: Cuts branches with visits ≥3 AND avg_reward <0.3 to eliminate unpromising paths
  - **Domain Scoring**: Each worker rewards quality reasoning (not just final answers):
    - Math: +0.30 notation, +0.25 steps, +0.20 symbolic validity, +0.16 conclusion
    - Code: +0.30 syntax, +0.25 documentation, +0.20 modularity, +0.16 correctness
    - Logic: +0.30 structure, +0.25 coherence, +0.20 premises, +0.16 conclusion
  - **Backpropagation**: Updates all ancestor nodes with rewards, enables informed future selection
  - **Best Path Extraction**: After N simulations (router-determined, default 10), selects highest-reward path
  - **Effect**: Finds non-obvious solutions by systematically exploring possibilities (like AlphaGo)

### ✅ Multi-Layer Verification

- **Symbolic Math Verification** (SymPy engine):
  - Converts candidates to symbolic expressions: `sympify("2*x + 3")`
  - Computes expected answer symbolically: `diff(x**2 + 3*x, x)`
  - Checks algebraic equivalence: `simplify(expected - candidate) == 0`
  - **Catches subtle errors**: "2x+3", "3+2x", "2(x+1.5)" all verify as equivalent
  - **Formal proof**: Not string matching, actual mathematical equivalence checking

- **Code Verification** (AST + Execution):
  - Parse to Abstract Syntax Tree: `ast.parse(code)`
  - Check syntax validity, detect dangerous patterns (eval, exec, __import__)
  - Language-specific validators (Python, JavaScript, TypeScript)
  - Optional sandboxed execution for runtime verification
  - **Effect**: Prevents malformed or unsafe code from passing

- **Semantic Verification** (Embedding-based):
  - Logic/Factual: Encode with sentence-transformers, measure coherence
  - Check conclusion presence, information completeness, specificity
  - Creative: Vocabulary diversity (unique words / total), sentence coherence
  - **Effect**: Validates reasoning quality beyond just surface correctness

### 🔄 Reflection Engine - Self-Correction Loop

- **Error Analysis**: When verification fails, systematically diagnose the issue:
  - Math: "Algebraic simplification error in step 3"
  - Code: "Syntax error on line 12: missing closing parenthesis"
  - Logic: "Conclusion doesn't follow from premises"
  
- **Reflection Prompting**: Generates enhanced context for retry:
  ```
  Previous attempt failed verification.
  Error: [specific issue identified]
  Key mistake: [detailed explanation]
  Correct approach: [guidance for improvement]
  
  Please provide corrected reasoning...
  ```

- **Iterative Retry**: Runs new LATS search with reflection guidance
  - Default: 2 iterations (configurable with `--max-reflection-iterations`)
  - Each iteration learns from previous mistakes
  - Stops early if verification passes
  - **Research basis**: Reflexion, Self-Refine papers show LLMs improve significantly with feedback

### 🎯 Quality-Aware Semantic Cache with LLM Validation

- **Two-Stage Validation** (Fast pre-filter + Intelligent validation):
  1. **Semantic Similarity Check** (0.001s):
     - Convert query to 384-dim embedding via sentence-transformers
     - Compute cosine similarity with all cached queries
     - Pre-filter: Only consider matches with similarity ≥ 0.85
     
  2. **LLM Validation Layer** (0.1-0.3s):
     - For similarity matches, ask reasoning LLM: "Would the cached answer FULLY and CORRECTLY satisfy the new query?"
     - Prompt includes: cached query, cached answer, new query
     - LLM responds: `{"valid": true/false, "confidence": 0.0-1.0, "reason": "..."}`
     - **Prevents false positives**: "integral of x²" vs "integral of x² from 0 to 1" have 0.89 similarity but different answers
     - LLM understands nuances that embeddings miss (definite vs indefinite integrals, boundary conditions, etc.)

- **Training Data Collection**:
  - Every validation decision logged to `.kaelum/cache_validation/validation_log.jsonl`
  - Format: `{timestamp, new_query, cached_query, cached_answer, validation_result}`
  - Export tool: `./export_cache_validation_data.py --output training.jsonl`
  - **Self-Improving System**: Collect validation data → Fine-tune validator → Deploy better model → Repeat

- **Quality Filtering**:
  - Only stores trees with quality="high" (successful verification + confidence ≥ 0.8)
  - Cache hits only return high-quality results (prevents serving incorrect cached answers)
  - Low-quality trees logged but never served
  - **Effect**: ~23% speedup on cache hits (0.001s vs 2-5s) with safety guarantees

- **Cache-First Architecture**:
  - Lookup happens BEFORE routing/detectors
  - Avoids unnecessary overhead on repeated queries
  - Cross-domain caching: Math solution can accelerate similar logic/analysis queries

### 🎚️ Adaptive Threshold Calibration

- **What are thresholds?**: Decision cutoffs for binary predictions (e.g., "Is this a math query?")
  - Model outputs confidence score (0-1)
  - Threshold determines cutoff: score > threshold → "yes"
  
- **Automated Optimization**:
  - Records (score, threshold, outcome) for every decision
  - After 20+ samples, runs grid search: tests thresholds [0.20, 0.25, ..., 0.85]
  - Calculates F1 score for each: `F1 = 2 * (precision × recall) / (precision + recall)`
  - Selects threshold that maximizes F1 (balances false positives vs false negatives)
  - Persists to `.kaelum/calibration/optimal_thresholds.json`
  
- **Graceful Degradation**:
  - Uses defaults when data insufficient
  - Per-domain calibration (math, code, logic, etc.)
  - **Effect**: System automatically tunes decision boundaries for best accuracy

### 📚 Active Learning & Fine-Tuning

- **Intelligent Training Example Selection**:
  - **Uncertainty sampling**: Queries where model had low confidence → improve weak areas
  - **Diversity sampling**: Max-min distance in embedding space → broad coverage
  - **Error mining**: Failed verifications with reflections → learn from mistakes
  - **Complexity selection**: High depth, many simulations, multi-step reasoning → train on hard problems
  - **Mixed strategy** (recommended): Balanced combination of all methods

- **Training Data Export**:
  - Automatic capture of (query, reasoning_steps, answer, metadata) during operation
  - Format: `{instruction, input, output}` for instruction-tuning
  - Export commands in `runtime/orchestrator.py`
  - Compatible with HuggingFace Transformers, OpenAI fine-tuning, etc.
  
- **Effect**: Continual learning loop - system generates its own training data from real usage

### 📊 Comprehensive Metrics & Analytics

- **Cache Statistics**: Hit rate, avg similarity, quality distribution, validation stats
- **Router Performance**: Worker selection accuracy, prediction errors, learning curves
- **LATS Metrics**: Avg tree depth, simulations per query, pruning efficiency, branch rewards
- **Verification Rates**: Pass/fail by domain, error types, reflection success rates
- **Token Tracking**: Input/output tokens per worker, cost estimation for cloud APIs
- **Latency Profiling**: Time spent in cache/routing/search/verification/reflection
- **Export**: JSON/CSV formats for external analysis and visualization

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start LLM Backend (vLLM Recommended)

```bash
# Install vLLM
pip install vllm

# Start server with a small fast model (recommended for testing)
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Or use a balanced model for production
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

### 4. Run Kaelum with Your Model

```bash
# Basic usage
python run.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# With custom reasoning parameters
python run.py --model microsoft/phi-4 \
    --base-url http://localhost:8000/v1 \
    --temperature 0.7 \
    --max-tree-depth 8 \
    --num-simulations 5

# See all options
python run.py --help
```

### 5. Docker Setup (Optional)

```bash
docker-compose up -d
```

---

## Supported LLMs

Kaelum is **model-agnostic** and works with any OpenAI-compatible API. Below are tested configurations optimized for reasoning tasks.

### 🚀 Recommended Models

| Model Family          | Size  | VRAM  | Speed  | Reasoning | Math/Code | Use Case                    | HuggingFace Model ID                              |
| --------------------- | ----- | ----- | ------ | --------- | --------- | --------------------------- | ------------------------------------------------- |
| **SmolLM2**     | 1.7B  | 3GB   | ⚡⚡⚡ | ⭐⭐⭐⭐  | ⭐⭐⭐    | Edge/Mobile, Fast inference | `HuggingFaceTB/SmolLM2-1.7B-Instruct`           |
| **Qwen 2.5**    | 3B    | 4GB   | ⚡⚡⚡ | ⭐⭐⭐⭐  | ⭐⭐⭐⭐  | Development, Testing        | `Qwen/Qwen2.5-3B-Instruct`                      |
| **Phi-3-mini**  | 3.8B  | 5GB   | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Strong reasoning, Low VRAM  | `microsoft/Phi-3-mini-4k-instruct`              |
| **Llama 3.2**   | 3B    | 4GB   | ⚡⚡⚡ | ⭐⭐⭐    | ⭐⭐⭐    | General purpose             | `meta-llama/Llama-3.2-3B-Instruct`              |
| **Qwen 2.5**    | 7B    | 8GB   | ⚡⚡   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best balance                | `Qwen/Qwen2.5-7B-Instruct`                      |
| **Llama 3.1**   | 8B    | 8GB   | ⚡⚡   | ⭐⭐⭐⭐  | ⭐⭐⭐⭐  | General reasoning           | `meta-llama/Llama-3.1-8B-Instruct`              |
| **DeepSeek-R1** | 7B    | 8GB   | ⚡⚡   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Math/Logic specialist       | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`      |
| **Phi-4**       | 14B   | 16GB  | ⚡     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex reasoning, SOTA     | `microsoft/phi-4`                               |
| **Qwen 2.5**    | 14B   | 16GB  | ⚡     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production quality          | `Qwen/Qwen2.5-14B-Instruct`                     |
| **Mixtral**     | 8x7B  | 24GB  | ⚡     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  | High quality, MoE           | `mistralai/Mixtral-8x7B-Instruct-v0.1`          |

**Key Highlights:**
- **SmolLM2-1.7B**: Smallest efficient model, excellent for edge deployment, on-device inference, and resource-constrained environments. Trained on 11T tokens with strong instruction following.
- **Phi-3-mini (3.8B)**: Microsoft's reasoning-optimized small model with exceptional math/logic performance (GSM8K: 85.7%, HumanEval: 57.3%). Best small model for reasoning.
- **Phi-4 (14B)**: Latest Microsoft model with SOTA small-model performance (MMLU: 84.8%, MATH: 80.4%, HumanEval: 82.6%). Best for complex reasoning tasks.
- **Qwen 2.5**: Strong all-around performance across sizes, excellent for code generation
- **DeepSeek-R1**: Specialized for mathematical and logical reasoning with reinforcement learning

### 🚀 vLLM Setup (Recommended)

**Best for:** Production deployments, high throughput, GPU optimization, batch processing

**Basic Setup:**

```bash
# 1. Install vLLM
pip install vllm

# 2. Start vLLM server (choose a model)
# Small & Fast (recommended for testing)
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Best Reasoning (recommended for production)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# High Quality (if you have VRAM)
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 3. Run Kaelum
python run.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
```

**Advanced vLLM Configuration:**

```bash
# Multi-GPU setup
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000

# Quantization for lower VRAM
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --quantization awq \
    --port 8000

# CPU offloading for large models
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --cpu-offload-gb 32 \
    --port 8000
```

**Supported Architectures:**
- Any Hugging Face model with chat template
- Qwen, Llama, Mistral, Yi, DeepSeek, Phi families
- Custom fine-tuned models with transformer architecture

### ☁️ Cloud APIs (Alternative)

| Provider              | Setup                                | Base URL                                  | Example                                                     |
| --------------------- | ------------------------------------ | ----------------------------------------- | ----------------------------------------------------------- |
| **OpenAI**      | Get API key from platform.openai.com | `https://api.openai.com/v1`             | `--model gpt-4 --base-url https://api.openai.com/v1`      |
| **Anthropic**   | Use with proxy/adapter               | Via proxy                                 | Use OpenAI-compatible proxy                                 |
| **Together AI** | Get key from together.ai             | `https://api.together.xyz/v1`           | `--model meta-llama/Llama-3-70b-chat-hf`                  |
| **Fireworks**   | Get key from fireworks.ai            | `https://api.fireworks.ai/inference/v1` | `--model accounts/fireworks/models/llama-v3-70b-instruct` |
| **Groq**        | Get key from groq.com                | `https://api.groq.com/openai/v1`        | `--model llama3-70b-8192`                                 |

**Example with OpenAI:**

```bash
export OPENAI_API_KEY="sk-..."
python run.py --model gpt-4 --base-url https://api.openai.com/v1
```

### 🏠 Other Deployment Options

| Option                          | Best For                        | Setup Difficulty | OpenAI Compatible |
| ------------------------------- | ------------------------------- | ---------------- | ----------------- |
| **vLLM** (Recommended)    | Production, GPU optimization    | ⭐⭐ Moderate    | ✅ Yes            |
| **Ollama**                | Quick local testing, beginners  | ⭐ Easy          | ✅ Yes            |
| **LM Studio**             | GUI-based, no-code deployment   | ⭐ Easy          | ✅ Yes            |
| **llama.cpp**             | CPU inference, low VRAM         | ⭐⭐ Moderate    | ✅ Yes (w/server) |
| **text-generation-webui** | Full UI + API                   | ⭐⭐ Moderate    | ✅ Yes            |
| **LocalAI**               | Docker-based multi-backend      | ⭐⭐ Moderate    | ✅ Yes            |

### 📊 Model Recommendations by Use Case

| Use Case                      | Recommended Model           | Why                                                |
| ----------------------------- | --------------------------- | -------------------------------------------------- |
| **Edge/Mobile**         | SmolLM2 1.7B                | Smallest efficient model, runs on-device           |
| **Development/Testing** | Qwen 2.5 3B / Phi-3-mini    | Fast inference, low VRAM, solid reasoning          |
| **Math/Logic**          | Phi-4 / DeepSeek-R1 7B      | Specialized for reasoning (Phi-4: 80.4% on MATH)   |
| **Code Generation**     | Qwen 2.5 14B / Phi-4        | Strong code capabilities, function calling support |
| **General Reasoning**   | Qwen 2.5 7B                 | Best balance of speed/quality/VRAM                 |
| **Production**          | Qwen 2.5 14B / Phi-4        | High quality, reliable, SOTA performance           |
| **Research**            | Custom fine-tuned           | Domain-specific optimization with PEFT/LoRA        |

---

## Detailed Setup Guide

### Step-by-Step: vLLM + Kaelum

This is the **recommended** way for production deployments:

```bash
# Step 1: Install vLLM
pip install vllm

# Step 2: Start vLLM with your chosen model
# For testing/development (low VRAM)
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# For production (balanced)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

# For high-quality reasoning
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# Step 3: Clone Kaelum
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Step 4: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Run with your vLLM model
python run.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# Step 7: (Optional) Customize reasoning settings
python run.py \
    --model microsoft/phi-4 \
    --base-url http://localhost:8000/v1 \
    --embedding-model all-mpnet-base-v2 \
    --temperature 0.7 \
    --max-tree-depth 8 \
    --num-simulations 20 \
    --enable-factual-verification \
    --debug-verification

# Step 8: See all options
python run.py --help
```

### Multi-GPU Setup with vLLM

For large models or high throughput:

```bash
# Tensor parallelism (split model across GPUs)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000

# Pipeline parallelism (for extremely large models)
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2 \
    --port 8000

# Quantization for VRAM optimization
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --quantization awq \
    --dtype half \
    --port 8000
```

### Alternative: Quick Testing with Ollama

For **quick local testing** without GPU setup complexity:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run a model
ollama pull Qwen/Qwen2.5-1.5B-Instruct
ollama serve

# Run Kaelum (in another terminal)
python run.py --model Qwen/Qwen2.5-1.5B-Instruct --base-url http://localhost:11434/v1
```

---

## Architecture Overview

### System Components

#### 1. **Core Pipeline** (`runtime/orchestrator.py`)

The orchestrator implements a carefully optimized query processing pipeline:

```
Query Input
    ↓
[1] Query Embedding (sentence-transformers, 384-dim)
    ↓
[2] Completeness Detection (checks if query is answerable)
    ↓
[3] CACHE LOOKUP (FIRST - before routing!)
    ├─ Similarity check (cosine ≥ 0.85)
    ├─ Quality filter (only high-quality trees)
    ├─ LLM validation (semantic correctness)
    └─ HIT? → Return cached result (0.001s) ✅
    ↓
[4] Detectors Run (cache miss only)
    ├─ Task Type (question/instruction/analysis/etc.)
    ├─ Worker Type (math/code/logic/factual/creative/analysis)
    └─ Domain Type (academic/technical/general/etc.)
    ↓
[5] Neural Router Decision
    ├─ Extract features: embedding + structural signals
    ├─ Forward pass: 398 → 256 → 128 → outputs
    ├─ Select: Best worker + tree depth + simulations
    └─ Log: Routing decision for learning
    ↓
[6] LATS Search (selected worker)
    ├─ Run N simulations (router-determined, default 10)
    ├─ UCT selection: Q/N + c×√(ln N_parent / N_node)
    ├─ Prune: visits ≥3 AND avg_reward <0.3
    ├─ Expand: LLM generates next reasoning steps
    ├─ Simulate: Score path with domain reward function
    ├─ Backpropagate: Update ancestors, check pruning
    └─ Extract: Best path (highest cumulative reward)
    ↓
[7] Verification
    ├─ Math: SymPy symbolic validation
    ├─ Code: AST parsing + syntax checks
    ├─ Logic/Factual: Semantic coherence + completeness
    ├─ Creative: Diversity + coherence metrics
    └─ PASS? → Go to [9], FAIL? → Go to [8]
    ↓
[8] Reflection (on verification failure)
    ├─ Analyze: Diagnose specific error type
    ├─ Generate: Reflection prompt with guidance
    ├─ Retry: New LATS search with reflection context
    └─ Iterate: Up to max_reflection_iterations (default 2)
    ↓
[9] Success Path
    ├─ Store: Cache tree with quality="high" + embedding
    ├─ Feedback: Enhanced router training data
        * (query, worker, success, avg_reward, depth, sims)
    ├─ Calibration: Update threshold statistics
    └─ Return: Final answer with metadata
```

**Key Design Decisions**:

- **Cache-first**: 23% speedup by checking cache before routing/detectors
- **Quality filtering**: Only serve verified high-confidence cached results
- **LLM validation**: Prevents false positives from embeddings alone
- **Enhanced feedback**: Router learns from rich signals (rewards, depth, sims) not just success/fail
- **Detector placement**: After cache to avoid overhead on cache hits
- **Early pruning**: Eliminates bad branches at visits=3 to save compute

#### 2. **Neural Router** (`core/router.py`)

**Architecture**:
```
Input: Query (text)
    ↓
Embedding: sentence-transformers → 384-dim vector
    ↓
Feature Extraction:
    - Query length (normalized)
    - Math symbols: ∂, ∫, √, ∑, ∏, etc. (count)
    - Code keywords: def, class, function, if, for, etc. (count)
    - Question words: what, how, why, when, where (binary)
    - Special tokens: quotes, brackets, operators (count)
    ↓
Concatenate: [384-dim embedding + 14-dim structural] → 398-dim
    ↓
Neural Network (PyTorch):
    Layer 1: Linear(398 → 256) + ReLU + Dropout(0.3)
    Layer 2: Linear(256 → 128) + ReLU + Dropout(0.3)
    ↓
Output Heads:
    Worker: Linear(128 → 6) + Softmax → probabilities for 6 workers
    Depth: Linear(128 → 1) + Sigmoid → [0,1] scaled to [3,10]
    Simulations: Linear(128 → 1) + Sigmoid → [0,1] scaled to [5,25]
```

**Training Process**:
1. **Data Collection**: Every query outcome stored:
   ```json
   {
     "query": "What is the derivative of x²?",
     "embedding": [0.23, -0.45, ...],
     "features": [12, 2, 0, ...],
     "worker": "math",
     "success": true,
     "avg_reward": 0.91,
     "actual_depth": 5,
     "actual_simulations": 10
   }
   ```

2. **Batch Formation** (after 32 outcomes):
   - Sample 16-32 diverse examples from buffer
   - Create tensors: `(query_features, worker_label, depth, sims)`

3. **Loss Computation**:
   ```python
   # Worker classification loss
   worker_loss = CrossEntropyLoss(predicted_probs, actual_worker)
   
   # Quality regression loss (predict avg_reward)
   quality_loss = MSELoss(predicted_quality, actual_avg_reward)
   
   # Depth/sims regression loss
   depth_loss = MSELoss(predicted_depth, actual_depth)
   sims_loss = MSELoss(predicted_sims, actual_simulations)
   
   # Combined loss
   total_loss = worker_loss + 0.5*quality_loss + 0.3*depth_loss + 0.3*sims_loss
   ```

4. **Optimization**: Adam optimizer with learning rate 0.001, gradient descent

5. **Persistence**: Save model weights to `.kaelum/router/policy_net.pth`

**Effect**: Router continuously improves worker selection accuracy and parameter predictions

#### 3. **LATS Implementation** (`core/lats.py`)

**Core Algorithm**:
```python
class LATSNode:
    query: str           # Current reasoning state
    parent: LATSNode     # Parent node
    children: List       # Child nodes
    visits: int = 0      # Times visited (N)
    total_reward: float = 0.0  # Cumulative reward (Q)
    is_pruned: bool = False    # Pruning flag
    
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def uct_score(self, c: float = 1.414) -> float:
        if self.is_pruned:
            return -inf  # Never select pruned nodes
        
        exploitation = self.avg_reward()
        exploration = c * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration
```

**Simulation Loop**:
```python
def run_simulation(self, root: LATSNode, worker: BaseWorker):
    # 1. Selection: Walk down tree using UCT
    node = root
    while node.children:
        # Filter out pruned nodes
        valid_children = [c for c in node.children if not c.is_pruned]
        if not valid_children:
            break
        # Select child with highest UCT score
        node = max(valid_children, key=lambda c: c.uct_score())
    
    # 2. Expansion: LLM generates next steps
    if node.visits > 0:  # Don't expand on first visit
        next_steps = worker.generate_next_steps(node.query)
        for step in next_steps:
            child = LATSNode(query=step, parent=node)
            node.children.append(child)
        node = random.choice(node.children)  # Select one to simulate
    
    # 3. Simulation: Score the reasoning path
    reward = worker.score_reasoning(node.query)
    
    # 4. Backpropagation: Update ancestors
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        
        # Early pruning check
        if node.visits >= 3 and node.avg_reward() < 0.3:
            node.is_pruned = True
        
        node = node.parent
```

**Worker-Specific Scoring Examples**:

```python
# Math Worker
def score_reasoning(self, text: str) -> float:
    score = 0.0
    if re.search(r'[∂∫√∑∏]|\\(frac|int|sum)', text):
        score += 0.30  # Mathematical notation
    if re.search(r'step \d+:|therefore|thus|hence', text, re.I):
        score += 0.25  # Step-by-step structure
    try:
        sp.sympify(extract_expression(text))
        score += 0.20  # Valid symbolic form
    except:
        pass
    if re.search(r'(answer|result|solution).*[:=]', text, re.I):
        score += 0.16  # Has conclusion
    return min(score, 1.0)

# Code Worker  
def score_reasoning(self, code: str) -> float:
    score = 0.0
    try:
        ast.parse(code)
        score += 0.30  # Valid syntax
    except:
        return 0.0  # Invalid code gets 0
    
    if re.search(r'(#|//|"""|\*/).*\w', code):
        score += 0.25  # Has comments
    if re.search(r'def |class |function ', code):
        score += 0.20  # Modular structure
    if 'return' in code:
        score += 0.16  # Returns result
    return min(score, 1.0)
```

**Pruning Impact**:
- Without pruning: 10 simulations explore ~8-10 paths uniformly
- With pruning: 10 simulations explore ~3-4 promising paths deeply
- Effect: 2-3x better solution quality at same compute budget

#### 4. **Tree Cache with LLM Validation** (`core/search/tree_cache.py`, `core/cache_validator.py`)

**Storage Format**:
```python
@dataclass
class CachedTree:
    query: str              # Original query
    query_embedding: ndarray  # 384-dim vector
    tree_id: str           # Unique identifier
    worker_specialty: str  # "math", "code", etc.
    created_at: float      # Timestamp
    success: bool          # Verification passed?
    confidence: float      # Model confidence (0-1)
    tree_path: str         # Path to full LATS tree JSON
```

**Lookup Process**:
```python
def get(self, query: str, query_embedding: ndarray) -> Optional[Dict]:
    # Stage 1: Fast similarity pre-filter
    best_match = None
    best_similarity = 0.0
    
    for cached_tree in self.cached_trees:
        if not cached_tree.success:  # Only consider successful trees
            continue
        
        similarity = cosine_similarity(query_embedding, cached_tree.query_embedding)
        if similarity < 0.85:  # Threshold filter
            continue
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cached_tree
    
    if best_match is None:
        return None  # No similar query found
    
    # Load cached result
    cached_data = load_json(best_match.tree_path)
    cached_answer = cached_data['result']['answer']
    
    # Stage 2: LLM validation
    validation = self.validator.validate_cache_match(
        new_query=query,
        cached_query=best_match.query,
        cached_answer=cached_answer
    )
    
    if not validation['valid']:
        # LLM says cached answer doesn't satisfy new query
        return None
    
    # Cache hit with LLM approval
    return cached_data
```

**LLM Validator** (`core/cache_validator.py`):
```python
class CacheValidator:
    def validate_cache_match(self, new_query, cached_query, cached_answer):
        prompt = f"""Analyze if a cached answer can be reused for a new query.

CACHED QUERY: {cached_query}
CACHED ANSWER: {cached_answer}

NEW QUERY: {new_query}

Question: Would the cached answer FULLY and CORRECTLY satisfy the new query?

Consider:
- Does the cached answer directly answer what the new query asks?
- Are there any constraints, conditions, or specifics in the new query 
  that the cached answer doesn't address?
- Would using this cached answer be misleading or incorrect?

Respond in JSON format:
{{"valid": true/false, "confidence": 0.0-1.0, "reason": "..."}}"""

        response = self.llm_client(prompt, temperature=0.1, max_tokens=200)
        result = json.loads(response)
        
        # Log for training
        self._log_validation(new_query, cached_query, cached_answer, result)
        
        return result
```

**Training Data Collection**:
- Every validation logged to `.kaelum/cache_validation/validation_log.jsonl`
- Export format for fine-tuning:
  ```json
  {
    "instruction": "Analyze if a cached answer can be reused for a new query.",
    "input": "CACHED QUERY: ...\nCACHED ANSWER: ...\n\nNEW QUERY: ...",
    "output": "{\"valid\": true, \"confidence\": 0.95, \"reason\": \"...\"}"
  }
  ```
- Export command: `./export_cache_validation_data.py --output training.jsonl`

**Why Two Stages?**:
- Stage 1 (similarity): Fast pre-filter using vector math (0.001s)
- Stage 2 (LLM): Slow but intelligent validation (0.1-0.3s)
- Combined: Best of both worlds - speed + accuracy

**Example Cases**:
| New Query | Cached Query | Similarity | LLM Valid? | Reason |
|-----------|--------------|------------|------------|---------|
| "integral of x²" | "derivative of x²" | 0.87 | ❌ False | Different operations (integral vs derivative) |
| "integral of x² from 0 to 1" | "integral of x²" | 0.91 | ❌ False | Definite vs indefinite - different answers |
| "find derivative of x squared" | "derivative of x²" | 0.93 | ✅ True | Same question, different phrasing |
| "what's d/dx of x²?" | "derivative of x² with respect to x" | 0.89 | ✅ True | Same question, mathematical notation |

#### 5. **Verification Engines** (`core/verification.py`, `core/sympy_engine.py`, `core/syntax_validator.py`)

**Math Verification** (SymPy):
```python
class MathVerification:
    def verify(self, query: str, candidate: str) -> bool:
        # Extract mathematical expressions
        expected_expr = self.extract_from_query(query)  # e.g., "derivative of x²+3x"
        candidate_expr = self.extract_from_answer(candidate)  # e.g., "2x + 3"
        
        try:
            # Parse to SymPy symbols
            x = sp.Symbol('x')
            expected = sp.diff(x**2 + 3*x, x)  # SymPy computes: 2*x + 3
            candidate_sym = sp.sympify(candidate_expr)  # Parse: 2*x + 3
            
            # Check algebraic equivalence
            difference = sp.simplify(expected - candidate_sym)
            return difference == 0
        except Exception as e:
            return False  # Fail-safe: reject if can't parse
```

**Code Verification** (AST):
```python
class CodeVerification:
    def verify(self, code: str, language: str = "python") -> bool:
        if language == "python":
            try:
                tree = ast.parse(code)
                
                # Check for dangerous patterns
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if node.names[0].name in ['os', 'sys', 'subprocess']:
                            return False  # Dangerous imports
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['eval', 'exec', '__import__']:
                                return False  # Code injection risks
                
                return True  # Valid and safe
            except SyntaxError:
                return False
        
        elif language == "javascript":
            # Use JS parser (e.g., esprima via py_mini_racer)
            try:
                parser.parse(code)
                return True
            except:
                return False
```

**Semantic Verification** (Logic/Factual):
```python
class SemanticVerification:
    def verify(self, query: str, answer: str) -> bool:
        # Encode with sentence transformer
        query_emb = self.encoder.encode(query)
        answer_emb = self.encoder.encode(answer)
        
        # Check semantic coherence
        coherence = cosine_similarity(query_emb, answer_emb)
        if coherence < 0.5:
            return False  # Answer unrelated to query
        
        # Check completeness
        if len(answer.split()) < 10:
            return False  # Too brief
        
        # Check conclusion presence
        if not re.search(r'(therefore|thus|hence|in conclusion)', answer, re.I):
            return False  # No clear conclusion
        
        # Check specificity (named entities, numbers, etc.)
        specificity = len(re.findall(r'\b[A-Z][a-z]+\b|\d+', answer))
        if specificity < 2:
            return False  # Too vague
        
        return True
```

#### 6. **Reflection Engine** (`core/reflection.py`)

**Error Analysis**:
```python
class ReflectionEngine:
    def analyze_failure(self, query: str, answer: str, error_type: str) -> str:
        analysis_prompts = {
            "math": """The mathematical answer failed symbolic verification.
                      Common issues: algebraic errors, wrong formula, computational mistakes.
                      Analyze where the reasoning went wrong.""",
            
            "code": """The code failed syntax validation.
                      Error details: {error}
                      Analyze the specific syntax issue and how to fix it.""",
            
            "logic": """The logical reasoning failed coherence checks.
                       The conclusion may not follow from the premises.
                       Analyze the logical gap."""
        }
        
        prompt = f"""Query: {query}
Previous Answer: {answer}
Error Type: {error_type}

{analysis_prompts[error_type]}

Provide specific error diagnosis:"""
        
        return self.llm_client(prompt, max_tokens=300)
```

**Reflection Prompt Generation**:
```python
def generate_reflection(self, query: str, previous_answer: str, error_analysis: str) -> str:
    return f"""You previously attempted to answer this question but made an error.

ORIGINAL QUERY: {query}

PREVIOUS ATTEMPT: {previous_answer}

ERROR ANALYSIS: {error_analysis}

KEY MISTAKES:
{self.extract_key_mistakes(error_analysis)}

CORRECT APPROACH:
{self.suggest_correct_approach(query, error_analysis)}

Please provide a corrected answer, carefully avoiding the previous mistakes.
Focus on {self.get_focus_area(error_analysis)}.
Show your step-by-step reasoning clearly."""
```

**Retry Loop**:
```python
def reflect_and_retry(self, query: str, max_iterations: int = 2):
    for iteration in range(max_iterations):
        # Run LATS search
        result = self.lats.search(query)
        
        # Verify
        verification = self.verify(query, result['answer'])
        if verification['passed']:
            return result  # Success!
        
        # Analyze failure
        error_analysis = self.analyze_failure(
            query, result['answer'], verification['error_type']
        )
        
        # Generate reflection
        reflection = self.generate_reflection(
            query, result['answer'], error_analysis
        )
        
        # Add reflection to context for next iteration
        query = f"{query}\n\n[REFLECTION FROM PREVIOUS ATTEMPT]\n{reflection}"
    
    # Max iterations reached
    return result  # Return last attempt even if unverified
```

**Effect**: ~40% improvement in eventual success rate through self-correction

---

## Configuration Options

All configuration is now via **command-line arguments** (no `.env` file needed):

```bash
python run.py --help
```

**Key Options:**

| Category               | Argument                          | Description              | Default                   |
| ---------------------- | --------------------------------- | ------------------------ | ------------------------- |
| **LLM**          | `--model`                       | Model name               | (required)                |
|                        | `--base-url`                    | API endpoint             | `http://localhost:8000/v1` |
|                        | `--temperature`                 | Creativity (0.0-2.0)     | `0.7`                   |
|                        | `--max-tokens`                  | Max response length      | `2048`                  |
| **Embeddings**   | `--embedding-model`             | Sentence transformer     | `all-MiniLM-L6-v2`      |
| **Search**       | `--max-tree-depth`              | LATS depth               | Router decides (3-10)     |
|                        | `--num-simulations`             | LATS simulations         | Router decides (5-25)     |
|                        | `--parallel`                    | Enable parallel search   | Disabled                  |
| **Routing**      | `--no-routing`                  | Disable neural router    | Enabled                   |
|                        | `--worker`                      | Force specific worker    | Auto                      |
| **Cache**        | `--no-cache`                    | Disable caching          | Enabled                   |
| **Verification** | `--no-symbolic-verification`    | Disable SymPy            | Enabled                   |
|                        | `--enable-factual-verification` | Enable fact checks       | Disabled                  |
| **Reflection**   | `--max-reflection-iterations`   | Self-correction attempts | `2`                     |
|                        | `--no-active-learning`          | Disable learning         | Enabled                   |

**Examples:**

```bash
# High accuracy mode (slower)
python run.py --model microsoft/phi-4 \
    --base-url http://localhost:8000/v1 \
    --max-tree-depth 10 \
    --num-simulations 25

# Fast mode (less accurate)
python run.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --base-url http://localhost:8000/v1 \
    --max-tree-depth 3 \
    --num-simulations 5

# Math-only with forced worker
python run.py --no-routing --worker math

# Debug mode
python run.py --debug-verification --enable-factual-verification

# Production settings
python run.py \
    --model qwen2.5:14b \
    --parallel \
    --max-workers 8 \
    --cache-dir /data/kaelum/cache
```

---

## Complete Workflow: From Query to Answer

This section walks through exactly what happens when you ask Kaelum a question, explaining both the **what** (steps) and **why** (concepts behind each step).

### Example Query: "What is the derivative of x² + 3x with respect to x?"

#### **Step 1: Query Embedding & Initial Cache Lookup**

```
Input: "What is the derivative of x² + 3x with respect to x?"
```

**What happens:**

- The system converts your question into a **384-dimensional embedding vector** using a sentence transformer model
- **Immediate cache check**: Compares query embedding with all cached successful solutions using cosine similarity
- If similarity ≥ 0.85 threshold AND quality="high", returns cached result instantly (0.001s)

**Why this matters:**
**Cache-first design** provides ~23% speedup by avoiding unnecessary routing and detector overhead. Only high-quality verified solutions are served from cache, preventing incorrect cached answers from being returned.

**Example:** If you previously asked "derivative of x²", the current query has ~0.91 similarity

- **Cache hit** → Return answer immediately (skip Steps 2-7)
- **Cache miss** → Continue to feature extraction and routing

#### **Step 2: Feature Extraction & Neural Router Selection**

**What happens (only if cache miss):**

- Router extracts **structural features**: query length, presence of math symbols (∂, ∫, √), code keywords, etc.
- These features are concatenated with the embedding into a **398-dimensional feature vector**
- The 398-dim feature vector goes through a **neural network** (2 hidden layers: 398→256→128)
- Network outputs:
  - **Worker probabilities**: [math: 0.92, logic: 0.04, code: 0.02, ...]
  - **Tree depth**: 5 (how deep to search)
  - **Simulations**: 10 (how many paths to explore)
- Selects "math" worker with 92% confidence

**Why this matters:**
The router is a **learned model** that improves over time using enhanced feedback (average tree rewards, actual depth, simulation counts). It trains on every query outcome using gradient descent. If it routes a calculus question to the "logic" worker and verification fails, it updates its weights to prefer "math" worker for similar queries. This is **continual learning** - the system gets smarter with use.

#### **Step 3: LATS - Monte Carlo Tree Search with Pruning**

```
Root: "derivative of x² + 3x"
 ├─ Node 1: "Apply power rule to x²" [Q=0.85, N=3]
 ├─ Node 2: "Use first principles" [Q=0.62, N=2] [PRUNED - low reward]
 └─ Node 3: "Apply sum rule first" [Q=0.91, N=5] ← Best path
```

**What happens (10 simulations):**

**Simulation 1-3: Initial Exploration**

- Start at root node
- LLM generates 3 possible first steps: "power rule", "first principles", "sum rule"
- Create child nodes for each option
- **Selection**: All untried, so explore each once

**Simulation 4-6: Exploitation with Early Pruning**

- For each node, calculate **UCT score**:
  ```
  UCT = (Total Reward / Visits) + 1.414 × √(ln(Parent Visits) / Node Visits)
         \_________________/       \___________________________________/
          Exploitation term          Exploration term
  ```
- **Pruning check**: If node has visits ≥ 3 AND average reward < 0.3, mark as pruned
- **"First principles" node**: Q=0.62, N=2 → continues exploring
- **"Sum rule" node** has Q=0.91, N=5 → UCT = 0.182 + 0.42 = 0.602
- **"Power rule" node** has Q=0.85, N=3 → UCT = 0.283 + 0.56 = 0.843 ← **Selected**
- LLM expands from "power rule" node: "d/dx(x²) = 2x, d/dx(3x) = 3"

**Simulation 7-10: Deep Exploitation**

- **"Sum rule" → individual derivatives → combine** path accumulates highest reward (0.91)
- This path gets selected more often (N=5 visits)
- **"First principles"** node accumulates 3 visits but avg_reward drops to 0.28 → **PRUNED** (no further exploration)
- Final reasoning: "Split into x² and 3x → derivatives are 2x and 3 → sum is 2x + 3"

**Scoring (Domain-Specific Reward Model):**
Each path is scored by the **MathWorker's reward function**:

- Contains mathematical notation: +0.30
- Shows step-by-step work: +0.25
- Valid symbolic form: +0.20 (checked with SymPy)
- Reaches conclusion: +0.16
- **Total reward**: 0.91

**Why this matters:**
MCTS balances **exploration** (trying new approaches) vs **exploitation** (following good paths). The UCT formula automatically handles this with **early pruning** to eliminate unpromising branches:

- High Q/N (exploitation): "This path worked well before"
- High exploration term: "We haven't tried this much yet"
- **Pruning**: "This path has been tried enough (≥3 visits) and performs poorly (<0.3 reward) - stop wasting simulations"

This is why AlphaGo beat world champions - MCTS finds non-obvious strategies by systematically exploring possibilities. For reasoning, it means considering multiple solution approaches before committing to one, while efficiently eliminating bad paths early.

#### **Step 4: Extract Best Path**

```
LATS tree → Traverse from root to leaf → Extract reasoning steps
Result: ["Apply sum rule", "d/dx(x²) = 2x", "d/dx(3x) = 3", "Combine: 2x + 3"]
```

**What happens:**

- After 10 simulations, select path with highest cumulative reward
- Extract the **sequence of reasoning steps** from root to leaf
- This becomes the candidate solution

#### **Step 5: Verification**

```
Candidate: "2x + 3"
SymPy verification: derivative(x**2 + 3*x, x) == 2*x + 3 →  TRUE
```

**What happens:**
The **MathWorker** uses **SymPy** (symbolic math engine) for verification:

```python
import sympy as sp
x = sp.Symbol('x')
expected = sp.diff(x**2 + 3*x, x)  # SymPy calculates: 2x + 3
candidate = sp.sympify("2*x + 3")   # Parse candidate
assert sp.simplify(expected - candidate) == 0  # Algebraically equivalent
```

**For other domains:**

- **Code**: AST parsing (check syntax validity)
- **Logic**: Semantic similarity + conclusion detection
- **Factual**: Information completeness + specificity scoring
- **Creative**: Vocabulary diversity + coherence metrics

**Why this matters:**
This isn't just checking if the answer "looks right" - it's **formal verification**. SymPy uses computer algebra to prove algebraic equivalence. "2x + 3" and "3 + 2x" and "2(x + 1.5)" all verify as correct because they're symbolically equivalent. This catches subtle errors that string matching would miss.

#### **Step 6: Success Path - Quality-Aware Cache Storage & Enhanced Router Feedback**

```
Verification passed
→ Store tree in cache with embedding + quality="high" metadata
→ Update router training data with enhanced feedback:
   {
     "query": "...",
     "worker": "math",
     "success": true,
     "avg_reward": 0.91,
     "actual_depth": 5,
     "actual_simulations": 10
   }
→ Return result
```

**What happens:**

- Successful tree stored in **semantic cache** with query embedding AND quality metadata
- Cache only serves results with quality="high" on future lookups (prevents serving low-confidence answers)
- Router records enhanced feedback: worker type, success/failure, average tree reward, actual search depth used, and simulation count
- Threshold calibrators record: "Worker selection confidence 0.92 was correct"
- Return answer: "The derivative is **2x + 3**"

**Enhanced router learning:**
After 32 successful outcomes, router runs gradient descent with richer feedback:

```python
loss = CrossEntropyLoss(predicted_worker, actual_best_worker)
# Router also learns from avg_reward to prefer workers that generate high-quality trees
reward_loss = MSELoss(predicted_quality, actual_avg_reward)
total_loss = loss + 0.5 * reward_loss
optimizer.backward(total_loss)
optimizer.step()  # Update neural network weights
```

#### **Alternative: Verification Failure → Reflection**

```
 Verification failed
→ Store in cache with quality="low" (not served on future lookups)
→ Reflection Engine analyzes error
→ Generate improved reasoning
→ Retry (up to max_iterations)
```

**What happens if verification fails:**

**Example:** Candidate answer was "2x + x" (wrong)

1. **Error Analysis:**

   ```
   Error: Algebraic simplification incorrect
   Issue: Added x instead of constant 3
   ```
2. **Reflection Prompt:**

   ```
   The previous attempt had an error in algebraic simplification.
   Key mistake: confused the derivative of 3x with additional x term.
   Correct approach: d/dx(3x) = 3 (constant factor rule)

   Please provide corrected reasoning...
   ```
3. **Retry:**

   - LLM generates improved reasoning with reflection context
   - New LATS search with reflection guidance
   - Verify again
   - If still fails, repeat (up to `max_reflection_iterations`, default 2)

**Why this matters:**
This is **self-correction** through reflection. The system doesn't just fail - it analyzes **why** it failed and tries again with that knowledge. Research shows LLMs significantly improve when given feedback about their mistakes (Reflexion, Self-Refine papers). Kaelum automates this process.

### Key Concepts Summary

| Concept                         | What It Does                                      | Why It Matters                                                     |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| **Embeddings**            | Convert text to vectors that capture meaning      | Enables semantic similarity, not just keyword matching             |
| **Neural Router**         | Learned model that selects expert worker          | Improves over time via gradient descent on outcomes                |
| **MCTS (UCT)**            | Explores multiple solution paths before deciding  | Finds non-obvious solutions by balancing exploration/exploitation  |
| **Domain Scoring**        | Rewards reasoning quality (not just final answer) | Prefers paths with clear logic, even if answer is partial          |
| **Symbolic Verification** | Formal proof of correctness (e.g., SymPy)         | Catches subtle errors that string matching misses                  |
| **Semantic Cache**        | Stores solutions with meaning-based lookup        | 1000x speedup on similar queries with natural language flexibility |
| **Reflection**            | Self-correction by analyzing failures             | Learns from mistakes like humans do                                |
| **Continual Learning**    | Router + thresholds improve with each query       | System gets smarter over time without manual retraining            |

### Performance Profile

```
Cache Hit:     0.001s  (semantic lookup)
New Query:     2-5s    (LATS + verification)
With Retry:    4-12s   (reflection + re-search)
```

The workflow is designed for **quality over speed** on first attempt, but **speed over recomputation** on similar queries. This makes Kaelum ideal for:

- Interactive problem-solving (tutoring, coding assistants)
- Repeated similar queries (documentation Q&A, support bots)
- Tasks requiring verified correctness (math, code generation)

---

## Example: Python API

```python
from kaelum import enhance, set_reasoning_model, get_metrics

# Optional: configure model / router settings
set_reasoning_model(
  base_url="http://localhost:11434/v1",
  model="Qwen/Qwen2.5-1.5B-Instruct",
  temperature=0.7,
  max_tokens=2048,
  enable_routing=True,
  use_symbolic_verification=True,
  max_reflection_iterations=2,
)

# Solve a query
result = enhance("What is the derivative of x^2 + 3x?")
print(result)

# Inspect metrics
metrics = get_metrics()
print(metrics["analytics"])
```

---

## Architecture Overview

Top-level components:

- core/router.py — PolicyNetwork: routes queries and predicts LATS depth and simulations.
- core/lats.py — LATS implementation (MCTS).
- core/workers.py & specialized workers — domain logic + prompts.
- core/verification.py — domain validators (SymPy, AST, embedding checks).
- core/tree_cache.py — semantic cache with cosine similarity lookup.
- core/reflection.py — error analysis and self-correction loop.
- runtime/orchestrator.py — pipeline orchestration and training data export.

Data flow:

1. Router embeds query and selects a worker + parameters.
2. Check global tree cache: if match (cosine > 0.85) return cached tree.
3. Run LATS (default 10 simulations) building multiple paths.
4. Verify candidate path with domain-specific rules.
5. If verification fails, reflection produces improved steps and retries (up to configured iterations).
6. Record outcomes for router training and threshold calibration.

---

## Verification & Reflection

Verification samples:

- Math: SymPy symbolic checks (equivalence, derivatives, integrals).
- Code: AST parse + language-specific checks (Python, JS, TS supported).
- Logic / Factual / Creative / Analysis: semantic checks, conclusion detection, specificity.

Reflection loop:

- Identify verification issues
- Generate revised reasoning steps
- Retry until pass or max iterations

---

## Adaptive Threshold Calibration

**What are thresholds?** In classification tasks (e.g., "Is this a math query?"), models output a confidence score (0-1). The threshold determines the cutoff - scores above it predict "yes", below predict "no".

How Kaelum optimizes thresholds:

- Records (confidence score, threshold used, whether prediction was correct) for every decision
- After sufficient samples (default 20), runs **grid search**: tests many threshold values (0.20, 0.25, ..., 0.85)
- Calculates **F1 score** for each threshold: `F1 = 2 * (precision * recall) / (precision + recall)`
  - **Precision**: Of predictions we made, how many were correct?
  - **Recall**: Of actual positives, how many did we find?
  - **F1 score**: Balances both metrics (1.0 = perfect, 0.0 = useless)
- Selects threshold that maximizes F1 score
- Persists optimal thresholds to `.kaelum/calibration/optimal_thresholds.json`
- Graceful fallback to default thresholds when data is insufficient

---

## LATS & UCT with Early Pruning

**What is UCT?** UCT (Upper Confidence Bound applied to Trees) is the selection algorithm that decides which path to explore next in the search tree. It balances exploitation (following promising paths) with exploration (trying untested options).

UCT formula:

```
UCT(node) = Q(node) / N(node) + c * sqrt(ln N(parent) / N(node))
```

- **Q(node)**: Cumulative reward from all simulations through this node (how good this path has been)
- **N(node)**: Visit count (how many times we've explored this node)
- **c**: Exploration constant (default √2) - higher values encourage more exploration
- **First term** (Q/N): Exploitation - prefer nodes with high average reward
- **Second term**: Exploration - prefer less-visited nodes to discover new paths

**Early pruning optimization:**

- Nodes with visits ≥ 3 AND average reward < 0.3 are marked as pruned
- Pruned nodes are excluded from UCT selection (no further exploration)
- Eliminates wasted simulations on low-quality reasoning paths
- Improves search efficiency and reduces latency

Default behavior:

- Simulations: 10 per query (router can increase for complex problems)
- Expand: LLM generates next reasoning steps from current node
- Simulate: Score the reasoning path using domain-specific reward functions
- Backpropagate: Update all ancestor nodes with the reward, check pruning threshold, helping future selection

---

## Tree Cache with Quality Filtering

**How it works:** The cache stores successful reasoning trees using semantic embeddings (vector representations that capture meaning, not just words). When a new query arrives, it's converted to an embedding and compared against cached queries. **Critically, only high-quality verified results are served from cache.**

- **Embeddings**: Generated via sentence-transformers (a neural network that converts text to fixed-length vectors)
- **Cosine similarity**: Measures how "close" two embeddings are in vector space (1.0 = identical, 0.0 = completely different)
- **Lookup threshold**: 0.85 (queries with similarity ≥ 0.85 check cache)
- **Quality filtering**: Cache only returns results marked with quality="high" (verified, high-confidence)
- **Cache-first design**: Lookup happens BEFORE routing/detectors for ~23% speedup on hits
- Successful trees stored with embeddings, quality metadata (high/low), confidence scores, and full reasoning trace
- **Cache hit**: Returns complete LATS tree instantly (~0.001s instead of 2-5s for new search)
- **Cache miss or low-quality**: Proceeds to routing, detectors, and full LATS search
- **Cross-domain caching**: A math solution can accelerate similar logic or analysis queries if semantically close

---

## Active Learning & Fine-Tuning

**What is active learning?** Instead of training on random data, intelligently select the most valuable examples (queries where the model struggled, diverse examples, complex reasoning, etc.) to maximize learning efficiency.

How Kaelum collects training data:

- Automatically captures (query, reasoning steps, answer) triples during operation
- **Selection strategies** for generating training batches:
  - **Uncertainty**: Queries where model had low confidence - helps improve weak areas
  - **Diversity**: Semantically diverse queries via max-min distance sampling - ensures broad coverage
  - **Error**: Failed verification attempts with reflection improvements - learns from mistakes
  - **Complexity**: High tree depth, many simulations, multi-step reasoning - trains on hard problems
  - **Mixed**: Balanced combination of all strategies (recommended)
- Export formatted datasets for fine-tuning with Hugging Face Transformers, OpenAI, etc.
- Fine-tuned models show improved performance on domain-specific reasoning tasks

---

## Testing & Development

```bash
pip install pytest pytest-cov
python -m pytest -v
python -m pytest --cov=core --cov=runtime
```

## Performance & Limits

- Default LATS simulations: 10 (router can increase for complex queries)
- Typical query latency: 2–5s (uncached); cached queries ~0.001s (1000x faster)
- Verification: High accuracy for math (SymPy symbolic validation) and Python AST parsing
- Language support: Python, JavaScript, TypeScript for code verification

---

### Common Issues

#### 1. **vLLM: Out of Memory (OOM) / CUDA out of memory**

**Problem:** Model too large for your GPU VRAM.

**Solutions:**

```bash
# Use a smaller model
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Reduce GPU memory utilization
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.7 \
    --port 8000

# Enable quantization (AWQ, GPTQ)
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --quantization awq \
    --port 8000

# Use CPU offloading for large models
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --cpu-offload-gb 16 \
    --port 8000

# Alternative: Use Ollama for easier memory management
ollama run Qwen/Qwen2.5-1.5B-Instruct
```

#### 2. **vLLM: Slow inference / Timeout errors**

**Problem:** Model inference is slow or timing out.

**Solutions:**

```bash
# Use smaller/faster model
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Enable tensor parallelism (multi-GPU)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000

# Reduce Kaelum search parameters
python run.py --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1 \
    --max-tree-depth 3 \
    --num-simulations 5

# Disable verification (faster but less accurate)
python run.py --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1 \
    --no-symbolic-verification

# Increase vLLM batch size for throughput
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 256 \
    --port 8000
```

#### 3. **vLLM: Model not found / Download errors**

**Problem:** vLLM can't find or download the model from Hugging Face.

**Solutions:**

```bash
# Verify model name is correct (case-sensitive)
# ✅ Correct: Qwen/Qwen2.5-7B-Instruct
# ❌ Wrong: qwen/qwen2.5-7b-instruct

# Pre-download model manually
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

# Use local model path
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/local/model \
    --port 8000

# Set HF token for gated models (Llama, etc.)
export HF_TOKEN="hf_..."
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

#### 4. **SymPy verification always fails**

**Problem:** Math expressions not in SymPy-compatible format.

**Solutions:**

```bash
# Disable symbolic verification if not needed
python run.py --no-symbolic-verification

# Check debug output
python run.py --debug-verification
```

#### 4. **Cache not working / Always computing fresh**

**Problem:** Cache disabled or similarity threshold too high.

**Solutions:**

```bash
# Ensure cache is enabled (it is by default)
python run.py  # Cache enabled

# Check cache directory exists
ls .kaelum/cache

# Lower similarity threshold (edit code if needed)
# Default is 0.85, can adjust in TreeCache class
```

#### 8. **Router always selects wrong worker**

**Problem:** Router needs training data.

**Solutions:**

```bash
# Force specific worker during testing
python run.py --worker math

# Disable router and use default
python run.py --no-routing

# Let router learn - it improves after ~10-20 queries
# Just keep using it!
```

### Performance Tuning

**For Maximum Accuracy (slower):**

```bash
python run.py \
    --model qwen2.5:14b \
    --max-tree-depth 10 \
    --num-simulations 25 \
    --max-reflection-iterations 5 \
    --enable-factual-verification
```

**For Maximum Speed (less accurate):**

```bash
python run.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-tree-depth 3 \
    --num-simulations 5 \
    --max-reflection-iterations 0 \
    --no-symbolic-verification
```

**Balanced (recommended):**

```bash
python run.py \
    --model qwen2.5:7b \
    --temperature 0.7
# Let router decide depth/sims automatically
```

## Research & References

Kaelum builds upon several key research areas in AI and reasoning:

- [Browne et al. (2012): &#34;A Survey of Monte Carlo Tree Search Methods&#34;](https://ieeexplore.ieee.org/document/6145622)
- [Silver et al. (2016): &#34;Mastering the game of Go with deep neural networks and tree search&#34; (AlphaGo)](https://www.nature.com/articles/nature16961)
- [Wei et al. (2022): &#34;Chain-of-Thought Prompting Elicits Reasoning in Large Language Models&#34;](https://arxiv.org/abs/2201.11903)
- [Yao et al. (2023): &#34;Tree of Thoughts: Deliberate Problem Solving with Large Language Models&#34;](https://arxiv.org/abs/2305.10601)
- [Shinn et al. (2023): &#34;Reflexion: Language Agents with Verbal Reinforcement Learning&#34;](https://arxiv.org/abs/2303.11366)
- [Madaan et al. (2023): &#34;Self-Refine: Iterative Refinement with Self-Feedback&#34;](https://arxiv.org/abs/2303.17651)
- [Shazeer et al. (2017): &#34;Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer&#34;](https://arxiv.org/abs/1701.06538)
- [Fedus et al. (2021): &#34;Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity&#34;](https://arxiv.org/abs/2101.03961)
- [Welleck et al. (2022): &#34;Symbolic Knowledge Distillation: from General Language Models to Commonsense Models&#34;](https://arxiv.org/abs/2110.07178)
- [Settles (2009): &#34;Active Learning Literature Survey&#34;](https://minds.wisconsin.edu/handle/1793/60660)
- [Reimers &amp; Gurevych (2019): &#34;Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks&#34;](https://arxiv.org/abs/1908.10084)
