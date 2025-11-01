"""MCP Runtime Orchestrator - Streamlined reasoning acceleration."""

import asyncio
from typing import Dict, List, Optional

from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.verification import VerificationEngine


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

    def infer(self, query: str) -> Dict:
        """Run reasoning acceleration pipeline."""
        # Generate initial reasoning trace
        trace = self.generator.generate_reasoning(query)
        initial_confidence = self._quick_confidence_check(trace)
        
        # Skip reflection if confidence is high
        if initial_confidence > 0.85:
            verification = self.verification.verify_trace(trace)
            # Generate final answer from trace
            answer = self.generator.generate_answer(query, trace)
            result = {
                "query": query,
                "final": answer,
                "trace": trace,
                "diagnostics": {
                    "confidence": initial_confidence,
                    "verification": verification,
                    "iterations": 0,
                },
            }
        else:
            # Run reflection to improve reasoning
            reflection = self.reflection.enhance_reasoning(query, trace)
            verification = self.verification.verify_trace(reflection["final_trace"])
            result = {
                "query": query,
                "final": reflection["final_answer"],
                "trace": reflection["final_trace"],
                "diagnostics": {
                    "confidence": reflection["final_verification"]["confidence"],
                    "verification": verification,
                    "iterations": len(reflection["iterations"]),
                },
            }

        return result
    
    def _quick_confidence_check(self, trace: List[str]) -> float:
        """Quick heuristic confidence check - optimized to skip reflection by default."""
        if not trace:
            return 0.0
        
        # Start high to avoid unnecessary reflection (speed optimization)
        confidence = 0.85
        
        # Boost for structured reasoning
        if len(trace) >= 2:
            confidence += 0.05
        
        # Boost for math/logic indicators
        trace_text = "".join(trace).lower()
        if any(char in trace_text for char in ["=", "+", "-", "×", "÷", "%"]):
            confidence += 0.05
        
        # Only reduce if we see uncertainty markers
        uncertainty_markers = ["maybe", "might", "unsure", "not sure", "unclear", "probably"]
        if any(marker in trace_text for marker in uncertainty_markers):
            confidence -= 0.15
        
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
