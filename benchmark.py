"""Ablation benchmark — evaluates Kaelum against standard baselines on GSM8K.

Configurations tested
---------------------
baseline   Direct single-shot LLM call; no search, no routing, no verification.
cot        Chain-of-thought prompt; single LLM call, no MCTS.
no_router  Kaelum LATS pipeline but routing disabled (always logic worker).
full       Kaelum full pipeline: neural routing + LATS + verification + reflection.

Metrics reported per config
---------------------------
- answer_accuracy   : exact numerical match against GSM8K ground truth
- avg_latency_s     : mean wall-clock time per query
- avg_llm_calls     : estimated number of LLM completions per query (from metadata)
- verification_pass : fraction of answers that passed the verification engine
- cache_hit_rate    : fraction of queries served from semantic cache

Usage
-----
    # Full run (100 questions, all 4 configs)
    python benchmark.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct

    # Quick smoke-test (10 questions, two configs)
    python benchmark.py --n 10 --configs baseline,full

    # Save CSV results
    python benchmark.py --output results/gsm8k_ablation.csv
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── dependency checks ─────────────────────────────────────────────────────────

def _require(pkg: str, install_hint: str = "") -> None:
    try:
        __import__(pkg)
    except ImportError:
        hint = f"  pip install {install_hint or pkg}"
        print(f"[benchmark] Missing dependency: {pkg}\n{hint}", file=sys.stderr)
        sys.exit(1)


# ── answer extraction ─────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_number(text: str) -> Optional[str]:
    """Extract the last number from a string (handles commas, decimals)."""
    text = text.replace(",", "").strip()
    # GSM8K final answers are always preceded by ####
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = _NUMBER_RE.search(after)
        if m:
            return m.group()
    # Fall back to last number in text
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def _answers_match(predicted: str, ground_truth: str) -> bool:
    p = _extract_number(predicted)
    g = _extract_number(ground_truth)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


# ── baseline implementations ──────────────────────────────────────────────────

def _run_baseline(llm, question: str) -> Tuple[str, Dict]:
    """Direct single-shot LLM call, no reasoning scaffolding."""
    from core.reasoning import Message
    t0 = time.time()
    resp = llm.generate([
        Message(role="system", content="You are a helpful math assistant. Answer concisely."),
        Message(role="user", content=question),
    ])
    return resp, {"latency": time.time() - t0, "llm_calls": 1,
                  "verification_passed": False, "cache_hit": False}


def _run_cot(llm, question: str) -> Tuple[str, Dict]:
    """Chain-of-thought prompt; single LLM call, no MCTS."""
    from core.reasoning import Message
    t0 = time.time()
    cot_prompt = (
        "Solve the following math problem step by step. "
        "Show your reasoning clearly, then state the final answer after '####'.\n\n"
        f"Problem: {question}"
    )
    resp = llm.generate([
        Message(role="system", content="You are an expert math reasoner."),
        Message(role="user", content=cot_prompt),
    ])
    return resp, {"latency": time.time() - t0, "llm_calls": 1,
                  "verification_passed": False, "cache_hit": False}


def _run_kaelum(orchestrator, question: str) -> Tuple[str, Dict]:
    """Run the Kaelum reasoning pipeline and return answer + metadata."""
    t0 = time.time()
    result = orchestrator.infer(question)
    latency = time.time() - t0

    answer = result.get("answer", "")
    metrics = result.get("metrics", {})
    return answer, {
        "latency": latency,
        "llm_calls": metrics.get("num_simulations", 1) + metrics.get("iterations", 1),
        "verification_passed": result.get("verification_passed", False),
        "cache_hit": result.get("cache_hit", False),
    }


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_gsm8k(n: int) -> List[Dict]:
    """Load first `n` examples from GSM8K test split via HuggingFace datasets."""
    _require("datasets", "datasets")
    from datasets import load_dataset  # type: ignore
    print(f"[benchmark] Loading GSM8K test split (n={n})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in ds.select(range(min(n, len(ds)))):
        examples.append({
            "question": row["question"],
            "answer":   row["answer"],           # contains #### <number>
        })
    print(f"[benchmark] Loaded {len(examples)} examples")
    return examples


# ── evaluation loop ───────────────────────────────────────────────────────────

def _evaluate_config(
    config_name: str,
    examples: List[Dict],
    runner,
    verbose: bool = False,
) -> Dict:
    correct = 0
    total_latency = 0.0
    total_llm_calls = 0
    total_verified = 0
    total_cache_hits = 0

    for i, ex in enumerate(examples):
        predicted, meta = runner(ex["question"])
        match = _answers_match(predicted, ex["answer"])
        if match:
            correct += 1
        total_latency += meta["latency"]
        total_llm_calls += meta["llm_calls"]
        total_verified += int(meta["verification_passed"])
        total_cache_hits += int(meta["cache_hit"])

        if verbose:
            status = "✓" if match else "✗"
            print(
                f"  [{config_name}] {i+1:3d}/{len(examples)} {status}  "
                f"({meta['latency']:.2f}s)"
            )

    n = len(examples)
    return {
        "config": config_name,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "avg_latency_s": total_latency / n if n else 0.0,
        "avg_llm_calls": total_llm_calls / n if n else 0.0,
        "verification_pass_rate": total_verified / n if n else 0.0,
        "cache_hit_rate": total_cache_hits / n if n else 0.0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaelum ablation benchmark on GSM8K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of GSM8K test questions to evaluate (default: 100)")
    parser.add_argument("--configs", default="baseline,cot,no_router,full",
                        help="Comma-separated list of configs to run")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM / OpenAI-compatible base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None,
                        help="Path to write CSV results (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results")
    args = parser.parse_args()

    configs_to_run = [c.strip() for c in args.configs.split(",")]
    valid_configs = {"baseline", "cot", "no_router", "full"}
    for c in configs_to_run:
        if c not in valid_configs:
            print(f"[benchmark] Unknown config '{c}'. Valid: {valid_configs}", file=sys.stderr)
            sys.exit(1)

    # ── load dataset ──────────────────────────────────────────────────────────
    examples = _load_gsm8k(args.n)

    # ── build shared LLM client ───────────────────────────────────────────────
    from core.config import KaelumConfig, LLMConfig
    from core.reasoning import LLMClient

    llm_cfg = LLMConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=0.0,   # greedy for reproducibility
        max_tokens=512,
    )
    llm = LLMClient(llm_cfg)

    # ── build kaelum orchestrators (full / no_router) ─────────────────────────
    def _make_orchestrator(enable_routing: bool):
        from kaelum import set_reasoning_model, _ensure_orchestrator
        # Re-init the singleton with the right settings
        set_reasoning_model(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=0.0,
            max_tokens=512,
            enable_routing=enable_routing,
        )
        return _ensure_orchestrator()

    # ── dispatch runners ──────────────────────────────────────────────────────
    all_results = []

    for cfg in configs_to_run:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg.upper()}")
        print(f"{'='*60}")

        if cfg == "baseline":
            runner = lambda q, _llm=llm: _run_baseline(_llm, q)
        elif cfg == "cot":
            runner = lambda q, _llm=llm: _run_cot(_llm, q)
        elif cfg == "no_router":
            orc = _make_orchestrator(enable_routing=False)
            runner = lambda q, _orc=orc: _run_kaelum(_orc, q)
        elif cfg == "full":
            orc = _make_orchestrator(enable_routing=True)
            runner = lambda q, _orc=orc: _run_kaelum(_orc, q)
        else:
            continue

        result = _evaluate_config(cfg, examples, runner, verbose=args.verbose)
        all_results.append(result)

        # Per-config summary
        print(f"\n  accuracy          : {result['accuracy']:.1%}  ({result['correct']}/{result['n']})")
        print(f"  avg latency       : {result['avg_latency_s']:.2f}s")
        print(f"  avg LLM calls     : {result['avg_llm_calls']:.1f}")
        print(f"  verification pass : {result['verification_pass_rate']:.1%}")
        print(f"  cache hit rate    : {result['cache_hit_rate']:.1%}")
        if result["errors"]:
            print(f"  errors            : {result['errors']}")

    # ── final comparison table ────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        header = f"{'Config':<12} {'Accuracy':>10} {'Latency':>10} {'LLM calls':>10} {'Verif%':>8} {'Cache%':>8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(
                f"{r['config']:<12} "
                f"{r['accuracy']:>9.1%} "
                f"{r['avg_latency_s']:>9.2f}s "
                f"{r['avg_llm_calls']:>10.1f} "
                f"{r['verification_pass_rate']:>7.1%} "
                f"{r['cache_hit_rate']:>7.1%}"
            )

    # ── write CSV ─────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_results[0].keys()) if all_results else []
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[benchmark] Results written to {out_path}")

    # Also emit JSON for programmatic consumption
    print("\n[benchmark] JSON results:")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
