"""Backend configuration and state management."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import DEFAULT_CACHE_DIR, DEFAULT_ROUTER_DIR

DEFAULT_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "api_key": None,
    "temperature": 0.7,
    "max_tokens": 512,
    "embedding_model": "all-MiniLM-L6-v2",
    "use_symbolic_verification": True,
    "use_factual_verification": False,
    "max_reflection_iterations": 2,
    "enable_routing": True,
    "parallel": False,
    "max_workers": 4,
    "router_learning_rate": 0.001,
    "router_buffer_size": 32,
    "router_exploration_rate": 0.1,
    "cache_dir": DEFAULT_CACHE_DIR,
    "router_data_dir": DEFAULT_ROUTER_DIR,
    "enable_active_learning": True,
}

WORKER_INFO = [
    {
        "name": "math",
        "description": "Mathematical reasoning with SymPy verification",
        "capabilities": ["calculus", "algebra", "equations", "symbolic math"],
        "verification": "symbolic"
    },
    {
        "name": "code",
        "description": "Code generation with AST validation",
        "capabilities": ["python", "javascript", "typescript", "syntax checking"],
        "verification": "ast_parsing"
    },
    {
        "name": "logic",
        "description": "Logical reasoning and argumentation",
        "capabilities": ["deduction", "premises", "conclusions", "coherence"],
        "verification": "semantic"
    },
    {
        "name": "factual",
        "description": "Fact-based questions with completeness checks",
        "capabilities": ["information retrieval", "specificity", "citations"],
        "verification": "semantic"
    },
    {
        "name": "creative",
        "description": "Creative writing and generation",
        "capabilities": ["stories", "ideas", "brainstorming", "diversity"],
        "verification": "coherence_diversity"
    },
    {
        "name": "analysis",
        "description": "Comprehensive analysis and evaluation",
        "capabilities": ["multi-perspective", "depth", "structured thinking"],
        "verification": "completeness"
    }
]
