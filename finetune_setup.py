"""
Fine-tuning setup for Kaelum workers.

This module provides utilities to prepare training data from reasoning traces
and fine-tune worker models on domain-specific tasks.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


@dataclass
class ReasoningTrace:
    """Single reasoning trace from LATS execution."""
    query: str
    domain: str  # math, code, logic, factual, creative, analysis
    trajectory: List[Dict]  # LATS tree nodes
    final_answer: str
    verified: bool
    reward: float
    metadata: Optional[Dict] = None


class ReasoningDataset(Dataset):
    """PyTorch dataset for reasoning traces."""
    
    def __init__(self, traces: List[ReasoningTrace], tokenizer, max_length: int = 1024):
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        # Format: Query + Chain of Thought + Answer
        prompt = f"Query: {trace.query}\n\nReasoning:\n"
        for step in trace.trajectory:
            prompt += f"- {step.get('thought', '')}\n"
        prompt += f"\nAnswer: {trace.final_answer}"
        
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


def collect_traces(log_dir: str = "./logs") -> List[ReasoningTrace]:
    """
    Collect reasoning traces from execution logs.
    
    Args:
        log_dir: Directory containing trace logs
    
    Returns:
        List of ReasoningTrace objects
    """
    traces = []
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return traces
    
    for trace_file in log_path.glob("*.json"):
        with open(trace_file) as f:
            data = json.load(f)
            trace = ReasoningTrace(
                query=data["query"],
                domain=data["domain"],
                trajectory=data["trajectory"],
                final_answer=data["final_answer"],
                verified=data["verified"],
                reward=data["reward"],
                metadata=data.get("metadata")
            )
            traces.append(trace)
    
    return traces


def filter_high_quality(traces: List[ReasoningTrace], 
                        min_reward: float = 0.7,
                        require_verified: bool = True) -> List[ReasoningTrace]:
    """
    Filter traces to only high-quality examples.
    
    Args:
        traces: All collected traces
        min_reward: Minimum reward threshold
        require_verified: Only include verified traces
    
    Returns:
        Filtered list of high-quality traces
    """
    filtered = []
    for trace in traces:
        if require_verified and not trace.verified:
            continue
        if trace.reward < min_reward:
            continue
        filtered.append(trace)
    
    return filtered


def prepare_dataset(traces: List[ReasoningTrace],
                   tokenizer,
                   domain: Optional[str] = None,
                   train_split: float = 0.9) -> Tuple[ReasoningDataset, ReasoningDataset]:
    """
    Prepare train/val datasets from traces.
    
    Args:
        traces: Reasoning traces
        tokenizer: HuggingFace tokenizer
        domain: Filter to specific domain (None = all domains)
        train_split: Train/val split ratio
    
    Returns:
        (train_dataset, val_dataset)
    """
    if domain:
        traces = [t for t in traces if t.domain == domain]
    
    split_idx = int(len(traces) * train_split)
    train_traces = traces[:split_idx]
    val_traces = traces[split_idx:]
    
    train_dataset = ReasoningDataset(train_traces, tokenizer)
    val_dataset = ReasoningDataset(val_traces, tokenizer)
    
    return train_dataset, val_dataset


def finetune_worker(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    domain: Optional[str] = None,
    output_dir: str = "./finetuned_models",
    log_dir: str = "./logs",
    min_reward: float = 0.7,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4
):
    """
    Fine-tune a worker model on collected reasoning traces.
    
    Args:
        model_name: Base model to fine-tune
        domain: Domain to specialize (None = all domains)
        output_dir: Where to save fine-tuned model
        log_dir: Directory with reasoning traces
        min_reward: Minimum reward for training examples
        epochs: Training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
    """
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Collecting traces from {log_dir}")
    traces = collect_traces(log_dir)
    print(f"Found {len(traces)} total traces")
    
    traces = filter_high_quality(traces, min_reward=min_reward)
    print(f"Filtered to {len(traces)} high-quality traces")
    
    if len(traces) == 0:
        raise ValueError("No high-quality traces found. Run Kaelum with logging enabled first.")
    
    train_dataset, val_dataset = prepare_dataset(traces, tokenizer, domain=domain)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Kaelum worker models")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model")
    parser.add_argument("--domain", default=None, help="Domain filter (math, code, logic, etc)")
    parser.add_argument("--output-dir", default="./finetuned_models", help="Output directory")
    parser.add_argument("--log-dir", default="./logs", help="Trace logs directory")
    parser.add_argument("--min-reward", type=float, default=0.7, help="Min reward threshold")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    finetune_worker(
        model_name=args.model,
        domain=args.domain,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        min_reward=args.min_reward,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
