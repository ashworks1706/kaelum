"""Reasoning plugin - wraps existing reasoning engine."""

from typing import Any, Dict, Optional
import time
from .base import KaelumPlugin


class ReasoningPlugin(KaelumPlugin):
    """Plugin for deep reasoning using local models."""
    
    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="reasoning", config=config)
        self.model_id = model_id
        self.base_url = base_url
        
        # Initialize LLM client and reasoning generator
        from kaelum.core.reasoning import LLMClient, ReasoningGenerator
        from kaelum.core.config import LLMConfig
        
        llm_config = LLMConfig(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        llm_client = LLMClient(llm_config)
        self.generator = ReasoningGenerator(
            llm_client=llm_client,
            system_prompt=system_prompt,
            user_template=user_template
        )
    
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Generate reasoning for input query."""
        start_time = time.time()
        
        # Extract query string
        query = input_data if isinstance(input_data, str) else input_data.get("query", "")
        
        # Generate reasoning trace
        reasoning_trace = self.generator.generate_reasoning(query, stream=False)
        
        # Generate final answer
        final_answer = self.generator.generate_answer(query, reasoning_trace, stream=False)
        
        # Combine into result
        result = f"{final_answer}\n\nReasoning:\n"
        for i, step in enumerate(reasoning_trace, 1):
            result += f"{i}. {step}\n"
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens = len(result.split())  # Rough estimate
        
        # Local model cost (negligible but track for completeness)
        cost = tokens * 0.00000001  # Effectively free vs commercial
        self.log_inference(tokens, latency_ms, cost)
        
        return result
    
    def get_cost_savings(self) -> float:
        """Calculate savings vs Gemini 2.0 Flash."""
        # Gemini 2.0 Flash: ~$0.10 per 1M tokens (blended)
        commercial_cost = (self._metrics.get("total_tokens", 0) / 1_000_000) * 0.10
        actual_cost = self._metrics.get("total_cost", 0)
        return commercial_cost - actual_cost
