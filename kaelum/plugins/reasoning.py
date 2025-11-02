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
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="reasoning", config=config)
        self.model_id = model_id
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.user_template = user_template
        
        # Initialize reasoning generator (lazy import to avoid circular deps)
        from kaelum.core.reasoning import ReasoningGenerator
        self.generator = ReasoningGenerator(
            model_id=model_id,
            base_url=base_url,
            system_prompt=system_prompt,
            user_template=user_template
        )
    
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Generate reasoning for input query."""
        start_time = time.time()
        
        # Extract query string
        query = input_data if isinstance(input_data, str) else input_data.get("query", "")
        
        # Generate reasoning
        result = self.generator.generate(
            query=query,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.7)
        )
        
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
