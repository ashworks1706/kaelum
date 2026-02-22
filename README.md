# Kaelum

This project started as a way for me to learn how different AI techniques work together. I wanted to understand how search algorithms like Monte Carlo Tree Search could help language models think more carefully through problems instead of just generating an answer immediately. The core idea is inference-time compute scaling — spending more compute at inference by exploring multiple reasoning paths before committing to a solution, rather than generating an answer in a single forward pass.

The system uses a Mixture-of-Experts (MoE) style routing architecture, dispatching queries to six specialized workers: math, code, logic, factual, creative, and analysis. Rewards come from a learned Process Reward Model (PRM) — a small MLP trained on verification outcomes that scores individual reasoning steps. Verification is handled by a fine-tunable HuggingFace text-classifier, not regex or heuristics. I added a semantic cache so identical questions don't need to be recomputed, and a neural router that learns which worker to use based on past performance via REINFORCE (reward-weighted policy gradient).

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

Here's the full path a query takes from your terminal to an answer.

1. Entry — [`kaelum.py`](kaelum.py)
The CLI parses your query, sets up the config, and hands everything to the orchestrator. It's also where streaming, metrics, and feedback submission come in.

2. Routing — [`core/search/router.py`](core/search/router.py)
The orchestrator embeds the query with `all-MiniLM-L6-v2` and feeds a PolicyNetwork (398→256 residual MLP with skip connection) the embedding plus length/keyword features. It outputs a softmax distribution over 6 workers, tree depth, and number of simulations. Epsilon-greedy exploration ($\varepsilon$-greedy: with probability $\varepsilon$ pick a random worker instead of the argmax) occasionally tries alternatives to avoid getting stuck. After the query resolves, the router trains one step via REINFORCE:

$$\mathcal{L}_{\text{router}} = \text{CrossEntropy}(\text{logits},\, y_{\text{worker}}) \cdot \bar{r}$$

where $y_{\text{worker}}$ is the correct worker index and $\bar{r}$ is the mean reward from the LATS simulation — so high-reward rollouts reinforce the routing choice more strongly. If confidence is low, an ensemble of workers can run in parallel and the highest-confidence result is chosen.

3. LATS search — [`core/search/lats.py`](core/search/lats.py) + [`core/search/reward_model.py`](core/search/reward_model.py) + [`core/verification/process_reward_model.py`](core/verification/process_reward_model.py)  
The chosen worker runs MCTS (Monte Carlo Tree Search — a planning algorithm that builds a tree of possible next steps and simulates many paths to figure out which direction looks most promising) over reasoning steps. Each node is a partial reasoning chain. At each step, the UCT formula picks which node to expand next:

$$\text{UCT}(s) = \underbrace{\frac{V(s)}{N(s)}}_{\text{exploit}} + C \cdot \underbrace{\sqrt{\frac{\ln N(\text{parent})}{N(s)}}}_{\text{explore}}$$

where $V(s)$ is the accumulated reward, $N(s)$ is how many times node $s$ has been visited, and $C = \sqrt{2} \approx 1.414$ is the exploration constant. If $N(s) = 0$ the score is $\infty$, so unvisited nodes are always tried first. The exploit term pushes the search toward branches that have scored well; the explore term pushes it toward branches that haven't been tried much yet.

Node rewards come from a learned Process Reward Model (1158→256→64→1 MLP, sigmoid output) that scores individual reasoning steps — not just whether the final answer was right, but whether each intermediate step was heading in the right direction. Its input is a concatenation of sentence embeddings:

$$\mathbf{f} = [\mathbf{q}_{384} \;\|\; \mathbf{s}_{384} \;\|\; \mathbf{c}_{384} \;\|\; \mathbf{w}_{6}] \in \mathbb{R}^{1158}$$

where $\mathbf{q}$, $\mathbf{s}$, $\mathbf{c}$ are the query, current step, and context embeddings, and $\mathbf{w}$ is a one-hot worker type vector. The PRM is trained online with MSE loss, activating after ≥50 samples and retraining every 25 new samples. After each simulation, `backpropagate()` walks up the tree adding the reward arithmetically: $V(s) \mathrel{+}= r$, $N(s) \mathrel{+}= 1$. Nodes whose average reward $V(s)/N(s)$ falls below the pruning threshold after enough visits get cut so later simulations don't waste time there.

4. Worker execution — [`core/workers/`](core/workers/)
Whichever worker was picked — math, code, logic, factual, creative, or analysis — runs the best LATS path through the LLM. Stopping is depth-based; rewards come from the PRM.

5. Verification + reflection — [`core/verification/verification.py`](core/verification/verification.py) + [`core/verification/reflection.py`](core/verification/reflection.py)
The answer goes through a learned-only verifier — a HuggingFace `pipeline("text-classification")` adapter (`LearnedVerifier`) whose pass/fail label is configurable. It maps the classifier's raw label to pass/fail by checking whether `"PASS"` (or your configured substring) appears in the label name. No regex, no heuristics, entirely data-driven — but that means it needs fine-tuning on your domain to be reliable out of the box. After that, a self-reflection loop has the LLM review its own output and either sign off or trigger a revision. If it fails, the existing LATS tree is reused rather than discarded — the failed path gets penalized and lightly-explored branches are un-pruned, so the next iteration continues MCTS from the same tree rather than restarting cold. Each reasoning step is recorded as a training example for the PRM.

6. Cache write-back + router update — [`core/search/tree_cache.py`](core/search/tree_cache.py), [`core/search/router.py`](core/search/router.py)
Results are stored, but retrieval is disabled (no heuristic gates). The router logs the outcome against its routing decision so it can update its weights over time.

7. Human feedback (RLHF) — [`core/learning/human_feedback.py`](core/learning/human_feedback.py)
You can rate the answer after the fact. `HumanFeedbackEngine` maintains a per-worker adjustment dict that is added to every PRM reward at inference time:

$$r_{\text{final}} = r_{\text{PRM}} + \delta_{\text{worker}}$$

When you mark a routing choice wrong, `_adjust_reward_models()` applies: $\delta_{\text{wrong}} \mathrel{-}= 0.03$ and $\delta_{\text{suggested}} \mathrel{+}= 0.05$. A wrong answer adds another $\delta_{\text{worker}} \mathrel{-}= 0.05$; a highly-rated answer ($\geq 4/5$) adds $\delta_{\text{worker}} \mathrel{+}= 0.02$. These persist to `reward_adjustments.json`, are loaded on startup, and directly lower or raise the reward signal LATS uses — so workers that have performed poorly on your queries get explored less by UCT. No gradient updates; just arithmetic deltas on the shared reward signal.

##### Legacy (No longer valid but left for reference):
<img width="1983" height="1098" alt="image" src="https://github.com/user-attachments/assets/97f5601e-e660-44b1-9338-80308e0d80d4" />
<img width="1983" height="915" alt="image" src="https://github.com/user-attachments/assets/1d810ebb-496f-494b-9f4a-cb3022dd22fe" />
<img width="1983" height="844" alt="image" src="https://github.com/user-attachments/assets/6b000d29-d8bc-4219-8157-de5bf966f229" />


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM (recommended)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7

# Or a smaller model for testing
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000
```

Then run queries directly from the CLI:

```bash
# Basic query
python kaelum.py "What is the integral of x^2?"

# Stream output token by token
python kaelum.py "Write a binary search in Python" --stream

# Hide reasoning trace, show answer only
python kaelum.py "Explain relativity" --no-trace

# Use a specific model or custom vLLM endpoint
python kaelum.py "Solve x^2 - 4 = 0" --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# Control search depth and simulations
python kaelum.py "Prove the Pythagorean theorem" --depth 5 --sims 20

# Output raw JSON
python kaelum.py "What is entropy?" --json

# Print session metrics
python kaelum.py --metrics

# Submit feedback for a past query
python kaelum.py --feedback "2+2?" --answer "4" --score 1.0
```

## Configuration

All options can be passed as CLI flags. The main ones:

| Flag | Default | Description |
|---|---|---|
| `--base-url` | `http://localhost:8000/v1` | vLLM / OpenAI-compatible endpoint |
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Model name |
| `--api-key` | — | API key if required |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `1024` | Max tokens per generation |
| `--depth` | per-worker default | Max LATS tree depth |
| `--sims` | per-worker default | Number of MCTS simulations |
| `--no-routing` | — | Disable neural router, use default worker |
| `--stream` | — | Stream tokens as they are generated |
| `--no-trace` | — | Hide reasoning trace |
| `--json` | — | Output raw JSON result |

## Project Structure

```
Kaelum/
├── kaelum.py          # CLI entry-point and library API
├── benchmark.py       # GSM8K ablation runner (baseline / CoT / no-router / full)
├── core/
│   ├── learning/      # Feedback and metrics
│   ├── search/        # LATS, router, reward model, tree cache
│   ├── verification/  # Learned verifier, PRM, reflection, confidence calibration
│   └── workers/       # Domain workers (math, code, logic, factual, creative, analysis)
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
- [Lightman et al. (2023): &#34;Let's Verify Step by Step&#34; (Process Reward Models)](https://arxiv.org/abs/2305.20050)
- [Settles (2009): &#34;Active Learning Literature Survey&#34;](https://minds.wisconsin.edu/handle/1793/60660)
- [Reimers &amp; Gurevych (2019): &#34;Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks&#34;](https://arxiv.org/abs/1908.10084)
