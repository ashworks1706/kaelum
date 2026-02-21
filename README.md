# Kaelum

This project started as a way for me to learn how different AI techniques work together. I wanted to understand how search algorithms like Monte Carlo Tree Search could help language models think more carefully through problems instead of just generating an answer immediately. The idea is to explore multiple reasoning paths before committing to a solution, kind of like how you might sketch out different approaches to a math problem before deciding which one works best.

The system routes queries to specialized workers depending on the question type. There's a math worker that uses symbolic verification with SymPy, a code worker that parses syntax trees, and workers for logic, factual questions, creative tasks, and analysis. Each one has different reward functions tuned for their domain. I added a semantic cache so identical questions don't need to be recomputed, and a neural router that learns which worker to use based on past performance.

The human feedback loop was something I added later when I realized the router could improve if users could tell it when it picked the wrong worker. Now you can rate the worker selection, answer quality, and individual reasoning steps. Those adjustments persist across sessions and actually influence future routing decisions and reward calculations.

---


## Metrics

Measured on `Qwen2.5-7B-Instruct`, 200 mixed queries across all worker types.

![Kaelum metrics chart](metrics.png)

| Metric | Baseline | Math | Code | Logic | Factual | Creative | Overall |
|---|---|---|---|---|---|---|---|
| Answer correctness | 61% | 84% | — | 79% | 76% | — | 80% |
| Syntax / structure validity | 78% | — | 97% | — | — | 83% | 93% |
| Verification pass (1st attempt) | — | 69% | 74% | 68% | 71% | 77% | 72% |
| Verification pass (after reflection) | — | 89% | 91% | 85% | 87% | 90% | 88% |
| Avg latency — cold (s) | 4.1 | 7.4 | 6.2 | 6.9 | 5.8 | 5.1 | 6.8 |
| Avg latency — cache hit (s) | 4.1 | 0.4 | 0.3 | 0.4 | 0.3 | 0.4 | 0.4 |
| Cache hit rate | — | 28% | 31% | 19% | 24% | 14% | 23% |
| Router accuracy (after 50 feedback samples) | — | 94% | 96% | 89% | 91% | 87% | 91% |

Cold latency is higher because LATS runs multiple simulations. The cache makes up for it on repeated or semantically similar queries.

---

## How It Works

When you send a query, the neural router picks which worker should handle it. That worker then uses LATS (Language Agent Tree Search) to build a reasoning tree where each node represents a reasoning step. The MCTS algorithm balances trying new approaches with following promising paths. Nodes that consistently give bad results get pruned to save compute. When the search finishes, the best path becomes the answer.

Workers have domain-specific reward functions - math heavily penalizes wrong answers, code checks syntax validity, creative writing looks for coherence and diversity. The cache stores good reasoning trees and reuses them for similar queries based on embedding similarity.

## Architecture

- **LATS**: MCTS with UCT formula for exploring reasoning paths
- **Neural Router**: Learns worker selection from outcomes and feedback
- **Workers**: Math (SymPy), Code (AST), Logic, Factual, Creative, Analysis
- **Semantic Cache**: Cosine similarity on embeddings with domain thresholds
- **Human Feedback**: Adjusts routing and rewards based on corrections


<img width="1983" height="1098" alt="image" src="https://github.com/user-attachments/assets/97f5601e-e660-44b1-9338-80308e0d80d4" />
<img width="1983" height="915" alt="image" src="https://github.com/user-attachments/assets/1d810ebb-496f-494b-9f4a-cb3022dd22fe" />
<img width="1983" height="844" alt="image" src="https://github.com/user-attachments/assets/6b000d29-d8bc-4219-8157-de5bf966f229" />


## Quick Start

### Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM (recommended)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7

# Or smaller model for testing
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Start backend
cd backend && python app.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Configuration

Copy `.env.example` to `.env`:

```bash
BACKEND_PORT=5000
LLM_BASE_URL=http://localhost:8000/v1
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Project Structure

```
Kaelum/
├── backend/           # Flask API
├── frontend/          # Next.js dashboard
├── core/
│   ├── detectors/     # Query classification
│   ├── learning/      # Feedback and metrics
│   ├── search/        # LATS, router, cache
│   ├── verification/  # SymPy, syntax validation
│   └── workers/       # Domain workers
└── runtime/           # Orchestrator
```


The hardest parts were getting the MCTS pruning right (too aggressive and you miss good paths, too lenient and you waste simulations) and tuning the domain-specific reward functions. They need to actually correlate with answer quality for the search to work properly.


---

## Papers Referenced

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
