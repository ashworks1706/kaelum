"""MCP Runtime Orchestrator."""

from typing import Dict, List
from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.verification import VerificationEngine


class MCP:
    """Modular Cognitive Processor."""

    def __init__(self, config: MCPConfig, rag_adapter=None):
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
        """Run reasoning pipeline."""
        trace = self.generator.generate_reasoning(query)
        answer = self.generator.generate_answer(query, trace)
        
        return {
            "query": query,
            "final": answer,
            "trace": trace,
        }
