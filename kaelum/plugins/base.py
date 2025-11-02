"""Base plugin interface for Kaelum extensions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class KaelumPlugin(ABC):
    """Base class for all Kaelum plugins."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._metrics = {"inferences": 0, "total_cost": 0.0, "total_latency_ms": 0.0}
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Process input and return result."""
        pass
    
    def log_inference(self, tokens: int, latency_ms: float, cost: float = 0.0):
        """Track inference metrics."""
        self._metrics["inferences"] += 1
        self._metrics["total_cost"] += cost
        self._metrics["total_latency_ms"] += latency_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return plugin metrics."""
        return {
            **self._metrics,
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / self._metrics["inferences"]
                if self._metrics["inferences"] > 0 else 0
            )
        }
    
    def get_cost_savings(self) -> float:
        """Calculate cost savings vs commercial LLM."""
        return 0.0
