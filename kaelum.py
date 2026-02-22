"""Kaelum — agentic LLM reasoning via LATS + neural routing.

Library API:
    from kaelum import enhance, enhance_stream, set_reasoning_model

CLI:
    python kaelum.py "your query"
    python kaelum.py "your query" --stream
    python kaelum.py "your query" --model Qwen/Qwen2.5-7B-Instruct --depth 5
    python kaelum.py --metrics
    python kaelum.py --feedback "Was that helpful?" --score 0.9
"""

__version__ = "2.1.0"

import sys
import json
import argparse
import textwrap
from typing import Optional, Dict, Any, Iterator

from core.config import KaelumConfig, LLMConfig
from runtime.orchestrator import KaelumOrchestrator
from core.paths import DEFAULT_CACHE_DIR, DEFAULT_ROUTER_DIR

# ─── singleton orchestrator ───────────────────────────────────────────────────

_orchestrator: Optional[KaelumOrchestrator] = None
_embedding_model: str = "all-MiniLM-L6-v2"


def set_reasoning_model(
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    api_key: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_reflection_iterations: int = 2,
    prm_pass_threshold: float = 0.5,
    enable_routing: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    router_data_dir: str = DEFAULT_ROUTER_DIR,
    parallel: bool = False,
    max_workers: int = 4,
    max_tree_depth: Optional[int] = None,
    num_simulations: Optional[int] = None,
    router_learning_rate: float = 0.001,
    router_buffer_size: int = 32,
    router_exploration_rate: float = 0.1,
    router_depth_min: int = 3,
    router_depth_max: int = 10,
    router_sims_min: int = 5,
    router_sims_max: int = 25,
    lats_exploration_constant: float = 1.414,
    lats_prune_visit_threshold: int = 3,
    lats_prune_reward_threshold: float = 0.3,
) -> None:
    """Initialize (or re-initialize) the reasoning orchestrator."""
    global _orchestrator, _embedding_model

    _embedding_model = embedding_model

    config = KaelumConfig(
        reasoning_llm=LLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        embedding_model=embedding_model,
        max_reflection_iterations=max_reflection_iterations,
        prm_pass_threshold=prm_pass_threshold,
        router_depth_min=router_depth_min,
        router_depth_max=router_depth_max,
        router_sims_min=router_sims_min,
        router_sims_max=router_sims_max,
        lats_exploration_constant=lats_exploration_constant,
        lats_prune_visit_threshold=lats_prune_visit_threshold,
        lats_prune_reward_threshold=lats_prune_reward_threshold,
    )

    _orchestrator = KaelumOrchestrator(
        config,
        enable_routing=enable_routing,
        cache_dir=cache_dir,
        router_data_dir=router_data_dir,
        parallel=parallel,
        max_workers=max_workers,
        max_tree_depth=max_tree_depth,
        num_simulations=num_simulations,
        router_learning_rate=router_learning_rate,
        router_buffer_size=router_buffer_size,
        router_exploration_rate=router_exploration_rate,
        router_depth_min=router_depth_min,
        router_depth_max=router_depth_max,
        router_sims_min=router_sims_min,
        router_sims_max=router_sims_max,
        lats_exploration_constant=lats_exploration_constant,
        lats_prune_visit_threshold=lats_prune_visit_threshold,
        lats_prune_reward_threshold=lats_prune_reward_threshold,
    )


def _ensure_orchestrator() -> KaelumOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        set_reasoning_model()
    return _orchestrator


# ─── library API ─────────────────────────────────────────────────────────────

def enhance(query: str) -> Dict[str, Any]:
    """Run query through the reasoning pipeline.

    Returns the raw result dict with keys:
        answer, reasoning_trace, worker, confidence,
        verification_passed, iterations, cache_hit
    """
    return _ensure_orchestrator().infer(query, stream=False)


def enhance_stream(query: str) -> Iterator[str]:
    """Stream tokens from the reasoning pipeline."""
    yield from _ensure_orchestrator().infer(query, stream=True)


def get_metrics() -> Dict[str, Any]:
    """Return session metrics from the orchestrator."""
    return _ensure_orchestrator().get_metrics_summary()


def submit_feedback(query: str, answer: str, score: float, notes: str = "") -> None:
    """Submit human feedback for a query/answer pair.

    Args:
        query:  The original query.
        answer: The answer that was given.
        score:  Quality score in [0, 1].
        notes:  Optional free-text notes.
    """
    orc = _ensure_orchestrator()
    orc.feedback_engine.record_feedback(query=query, answer=answer, score=score, notes=notes)


# ─── CLI rendering helpers ────────────────────────────────────────────────────

_WORKER_COLORS = {
    "math":     "\033[93m",   # yellow
    "code":     "\033[94m",   # blue
    "logic":    "\033[96m",   # cyan
    "factual":  "\033[92m",   # green
    "creative": "\033[95m",   # magenta
    "analysis": "\033[97m",   # bright white
    "general":  "\033[37m",   # grey
}
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"


def _worker_badge(worker: str) -> str:
    color = _WORKER_COLORS.get(worker.lower(), "\033[37m")
    return f"{color}{_BOLD}[{worker.upper()}]{_RESET}"


def _print_result(result: Dict[str, Any], *, show_trace: bool = True) -> None:
    """Pretty-print a reasoning result to stdout."""
    answer    = result.get("answer", "").strip()
    trace     = result.get("reasoning_trace", [])
    worker    = result.get("worker", "unknown")
    conf      = result.get("confidence", 0.0)
    verified  = result.get("verification_passed", False)
    iters     = result.get("iterations", 1)
    cache_hit = result.get("cache_hit", False)

    # ── header bar ──
    ver_str  = f"{_GREEN}verified{_RESET}" if verified else f"{_RED}✗ unverified{_RESET}"
    conf_str = f"{_YELLOW}{conf:.0%}{_RESET}"
    iter_str = f"{_DIM}×{iters}{_RESET}" if iters > 1 else ""
    cache_str = f" {_CYAN}⚡ cached{_RESET}" if cache_hit else ""
    print(f"\n{_worker_badge(worker)}  conf {conf_str}  {ver_str}{iter_str}{cache_str}\n")

    # ── reasoning trace ──
    if show_trace and trace:
        print(f"{_DIM}── reasoning ──────────────────────────────{_RESET}")
        for i, step in enumerate(trace, 1):
            wrapped = textwrap.fill(step.strip(), width=90,
                                    initial_indent=f"  {_DIM}{i}.{_RESET} ",
                                    subsequent_indent="     ")
            print(wrapped)
        print(f"{_DIM}───────────────────────────────────────────{_RESET}\n")

    # ── answer ──
    print(textwrap.fill(answer, width=90))
    print()


# ─── CLI entry-point ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kaelum",
        description="Kaelum — agentic LLM reasoning with LATS + neural routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python kaelum.py "What is the integral of x^2?"
              python kaelum.py "Write a binary search in Python" --stream
              python kaelum.py "Explain relativity" --no-trace
              python kaelum.py --metrics
              python kaelum.py --feedback "2+2?" --answer "4" --score 1.0
        """),
    )

    # ── query ──
    p.add_argument("query", nargs="?", help="Query to reason about")
    p.add_argument("--stream", action="store_true", help="Stream tokens as they are generated")
    p.add_argument("--no-trace", dest="no_trace", action="store_true",
                   help="Hide the reasoning trace, show answer only")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Output raw JSON result")

    # ── model config ──
    m = p.add_argument_group("model")
    m.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM / OpenAI-compatible base URL (default: %(default)s)")
    m.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Model name (default: %(default)s)")
    m.add_argument("--api-key", default=None, help="API key if required")
    m.add_argument("--temperature", type=float, default=0.7)
    m.add_argument("--max-tokens", type=int, default=1024)

    # ── search config ──
    s = p.add_argument_group("search")
    s.add_argument("--depth", type=int, default=None,
                   help="Max LATS tree depth (default: per-worker)")
    s.add_argument("--sims", type=int, default=None,
                   help="Number of MCTS simulations (default: per-worker)")
    s.add_argument("--no-routing", dest="no_routing", action="store_true",
                   help="Disable neural router, use default worker")

    # ── misc ──
    p.add_argument("--metrics", action="store_true",
                   help="Print session metrics and exit")
    p.add_argument("--feedback", metavar="QUERY",
                   help="Submit human feedback for a query")
    p.add_argument("--answer", metavar="ANSWER",
                   help="Answer text for --feedback")
    p.add_argument("--score", type=float, metavar="SCORE",
                   help="Quality score [0,1] for --feedback")
    p.add_argument("--notes", default="", metavar="NOTES",
                   help="Optional notes for --feedback")
    p.add_argument("--version", action="version", version=f"kaelum {__version__}")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # initialise orchestrator with CLI args
    set_reasoning_model(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_routing=not args.no_routing,
        max_tree_depth=args.depth,
        num_simulations=args.sims,
    )

    # ── --metrics ──
    if args.metrics:
        m = get_metrics()
        print(json.dumps(m, indent=2, default=str))
        return

    # ── --feedback ──
    if args.feedback is not None:
        if args.answer is None or args.score is None:
            parser.error("--feedback requires --answer and --score")
        submit_feedback(args.feedback, args.answer, args.score, args.notes)
        print(f"{_GREEN}Feedback recorded (score={args.score}){_RESET}")
        return

    # ── query ──
    if not args.query:
        parser.print_help()
        sys.exit(0)

    if args.stream:
        # streaming: print tokens directly then a trailing newline
        for chunk in enhance_stream(args.query):
            print(chunk, end="", flush=True)
        print()
        return

    result = enhance(args.query)

    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result, show_trace=not args.no_trace)


# ─── exports ─────────────────────────────────────────────────────────────────

__all__ = [
    "enhance",
    "enhance_stream",
    "set_reasoning_model",
    "get_metrics",
    "submit_feedback",
]

if __name__ == "__main__":
    main()
