"""MCP Runtime Orchestrator."""

from typing import Dict, List
from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.verification import VerificationEngine


class MCP:
    """Modular Cognitive Processor."""

    def __init__(self, config: MCPConfig, rag_adapter=None, reasoning_system_prompt=None, reasoning_user_template=None):
        self.config = config
        self.llm = LLMClient(config.reasoning_llm)
        self.generator = ReasoningGenerator(
            self.llm,
            system_prompt=reasoning_system_prompt,
            user_template=reasoning_user_template
        )
        self.reflection = ReflectionEngine(self.llm, max_iterations=config.max_reflection_iterations)
        self.verification = VerificationEngine(
            use_symbolic=config.use_symbolic_verification,
            use_factual_check=config.use_factual_verification,
            rag_adapter=rag_adapter
        )

    def infer(self, query: str, stream: bool = False):
        """Run reasoning pipeline."""
        if stream:
            return self._infer_stream(query)
        else:
            return self._infer_sync(query)
    
    def _infer_sync(self, query: str) -> Dict:
        """Non-streaming inference."""
        trace = self.generator.generate_reasoning(query)
        errors = self.verification.verify_trace(trace)
        
        if errors or self.config.max_reflection_iterations > 0:
            trace = self.reflection.enhance_reasoning(query, trace)
        
        answer = self.generator.generate_answer(query, trace)
        
        return {
            "query": query,
            "final": answer,
            "trace": trace,
        }
    
    def _infer_stream(self, query: str):
        """Streaming inference."""
        yield "🧠 Generating reasoning trace...\n\n"
        
        trace_text = ""
        for chunk in self.generator.generate_reasoning(query, stream=True):
            trace_text += chunk
            yield chunk
        
        yield "\n\n✓ Reasoning complete. Generating answer...\n\n"
        
        trace = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    trace.append(step)
        
        if not trace:
            trace = [trace_text.strip()]
        
        for chunk in self.generator.generate_answer(query, trace, stream=True):
            yield chunk
