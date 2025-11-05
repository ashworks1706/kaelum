# Fine-tuning Guide

## Overview

Kaelum supports fine-tuning worker models on domain-specific reasoning tasks using collected LATS execution traces. This improves worker performance on your specific use cases.

## Setup

### 1. Enable Trace Logging

Add to your code:

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator(enable_logging=True, log_dir="./logs")
```

### 2. Collect Training Data

Run Kaelum on diverse queries in your domain. The system will automatically log:
- LATS reasoning trajectories
- Verification results
- Reward signals

High-quality traces (verified + high reward) will be used for training.

### 3. Install Fine-tuning Dependencies

```bash
pip install transformers datasets accelerate peft bitsandbytes
```

## Fine-tuning

### Basic Usage

Fine-tune on all domains:

```bash
python finetune_setup.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --log-dir ./logs \
  --output-dir ./finetuned_models \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5
```

### Domain-Specific Fine-tuning

Specialize for one domain:

```bash
python finetune_setup.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --domain math \
  --output-dir ./finetuned_math \
  --min-reward 0.8
```

Supported domains: `math`, `code`, `logic`, `factual`, `creative`, `analysis`

### Advanced Options

```bash
python finetune_setup.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --domain code \
  --output-dir ./finetuned_code \
  --log-dir ./production_logs \
  --min-reward 0.85 \
  --epochs 5 \
  --batch-size 2 \
  --lr 1e-5
```

Parameters:
- `--min-reward`: Minimum reward threshold for training examples (default 0.7)
- `--epochs`: Number of training epochs (default 3)
- `--batch-size`: Batch size per GPU (default 4)
- `--lr`: Learning rate (default 2e-5)

## Using Fine-tuned Models

Update worker configuration to use fine-tuned models:

```python
from core.config import Settings

settings = Settings(
    MATH_WORKER_MODEL="./finetuned_models",  # or ./finetuned_math
    CODE_WORKER_MODEL="./finetuned_models",
    # ... other workers
)
```

## Training Data Quality

The system automatically filters traces:
- **Verified**: Must pass domain-specific verification
- **High reward**: Above `min_reward` threshold
- **Complete**: Full LATS trajectory recorded

Recommended minimum: 100+ high-quality traces per domain

## Memory Requirements

| Model Size | GPU Memory | Batch Size |
|------------|------------|------------|
| 3B params  | 16GB       | 4          |
| 7B params  | 24GB       | 2          |
| 13B params | 40GB       | 1          |

Use gradient accumulation for larger effective batch sizes with limited memory.

## Monitoring

Track training progress:

```bash
# View logs
tail -f finetuned_models/trainer_state.json

# Check eval loss
grep "eval_loss" finetuned_models/trainer_state.json
```

## Best Practices

1. **Diverse data**: Collect traces from varied queries
2. **High quality**: Use `min_reward >= 0.7` for training
3. **Domain-specific**: Fine-tune separate models per domain for best results
4. **Iterative**: Fine-tune → deploy → collect more data → fine-tune again
5. **Validation**: Test fine-tuned models on held-out queries before production

## Programmatic Usage

```python
from finetune_setup import (
    collect_traces,
    filter_high_quality,
    prepare_dataset,
    finetune_worker
)

# Custom fine-tuning pipeline
traces = collect_traces("./logs")
high_quality = filter_high_quality(traces, min_reward=0.8)

# Filter by custom criteria
math_traces = [t for t in high_quality if t.domain == "math" and t.reward > 0.9]

# Fine-tune with custom settings
finetune_worker(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    domain="math",
    output_dir="./specialized_math_model",
    min_reward=0.9,
    epochs=5
)
```

## Troubleshooting

**No traces found**: Enable logging in orchestrator before running queries

**Low quality traces**: Adjust verification thresholds or improve base model prompts

**OOM errors**: Reduce batch size or use gradient accumulation

**Poor fine-tuned performance**: Collect more diverse training data or increase `min_reward`
