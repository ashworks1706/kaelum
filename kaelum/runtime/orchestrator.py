"""MCP Runtime Orchestrator - Streamlined reasoning acceleration."""

import asyncio
from typing import Dict, List, Optional

from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.scoring import ConfidenceScorer
from kaelum.core.verification import VerificationEngine
from kaelum.core.cache import ReasoningCache


class MCP:
    """Modular Cognitive Processor - Accelerates LLM reasoning."""

    def __init__(self, config: MCPConfig, rag_adapter=None):
        """
        Initialize MCP with configuration.
        
        Args:
            config: MCP configuration
            rag_adapter: Optional RAG adapter for factual verification
        """
        self.config = config
        self.llm = LLMClient(config.llm)
        self.generator = ReasoningGenerator(self.llm)
        self.reflection = ReflectionEngine(self.llm, max_iterations=config.max_reflection_iterations)
        self.verification = VerificationEngine(
            use_symbolic=config.use_symbolic_verification,
            use_factual_check=config.use_factual_verification,
            rag_adapter=rag_adapter
        )
        self.scorer = ConfidenceScorer(symbolic_weight=0.3, factual_weight=0.3, verifier_weight=0.4)
        self.cache = ReasoningCache()

    def infer(self, query: str, use_cache: bool = True) -> Dict:
        """Run reasoning acceleration pipeline."""
        if use_cache:
            cached = self.cache.get(query, str(self.config))
            if cached:
                cached["cache_hit"] = True
                return cached

        initial = self.generator.generate_reasoning(query)
        initial_confidence = self._quick_confidence_check(initial["trace"])
        
        if initial_confidence > 0.85 and self.config.max_reflection_iterations == 1:
            verification = self.verification.verify_trace(initial["trace"])
            confidence = self.scorer.compute_confidence(
                symbolic_score=0.8 if verification.get("verified") else 0.3,
                factual_score=0.8 if verification.get("verified") else 0.3,
                verifier_confidence=initial_confidence,
            )
            result = {
                "query": query,
                "trace": initial["trace"],
                "confidence": confidence,
                "verification": verification,
                "iterations": 0,
                "improved": False,
                "cache_hit": False,
            }
        else:
            reflection = self.reflection.enhance_reasoning(query, initial["trace"])
            verification = self.verification.verify_trace(reflection["final_trace"])
            confidence = self.scorer.compute_confidence(
                symbolic_score=0.8 if verification.get("verified") else 0.3,
                factual_score=0.8 if verification.get("verified") else 0.3,
                verifier_confidence=reflection["final_verification"]["confidence"],
            )
            result = {
                "query": query,
                "trace": reflection["final_trace"],
                "confidence": confidence,
                "verification": verification,
                "iterations": len(reflection["iterations"]),
                "improved": reflection["improved"],
                "cache_hit": False,
            }
        
        if use_cache and confidence >= self.config.confidence_threshold:
            self.cache.set(query, str(self.config), result)

        return result
    
    def _quick_confidence_check(self, trace: List[str]) -> float:
        """Quick heuristic confidence check."""
        if not trace:
            return 0.0
        confidence = 0.7
        if len(trace) >= 3:
            confidence += 0.1
        if any(char in "".join(trace) for char in ["=", "+", "-", "×", "÷", "%"]):
            confidence += 0.1
        return min(confidence, 1.0)


class ModelRuntime:
    """Runtime wrapper for MCP integration."""

    def __init__(self, mcp: MCP):
        self.mcp = mcp

    def __call__(self, query: str) -> str:
        result = self.mcp.infer(query)
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(result["trace"]))
        return f"""Reasoning:
{trace_text}

Confidence: {result["confidence"]:.2f}
Iterations: {result["iterations"]}
"""
