"""MCP Runtime Orchestrator."""

from typing import Dict, List
from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.verification import VerificationEngine


class MCP:
    """Modular Cognitive Processor - uses YOUR reasoning model for all operations."""

    def __init__(self, config: MCPConfig, rag_adapter=None):
        self.config = config
        
        # YOUR reasoning LLM does all the work
        self.reasoning_llm = LLMClient(config.reasoning_llm)
        
        # All components use YOUR reasoning model
        self.generator = ReasoningGenerator(self.reasoning_llm)
        self.reflection = ReflectionEngine(self.reasoning_llm, max_iterations=config.max_reflection_iterations)
        self.verification = VerificationEngine(
            use_symbolic=config.use_symbolic_verification,
            use_factual_check=config.use_factual_verification,
            rag_adapter=rag_adapter
        )

    def infer(self, query: str) -> Dict:
        """Run reasoning pipeline using YOUR reasoning model."""
        trace = self.generator.generate_reasoning(query)
        answer = self.generator.generate_answer(query, trace)
        
        return {
            "query": query,
            "final": answer,
            "trace": trace,
        }
