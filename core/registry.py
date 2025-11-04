"""Model registry for domain-specific and specialized models."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    model_id: str
    model_type: str  # reasoning, planning, routing, vision
    base_url: str
    description: str
    domain: Optional[str] = None  # e.g., "medical", "legal", "code"
    context_length: int = 4096
    quantization: Optional[str] = None  # e.g., "4bit", "8bit"
    vram_gb: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self._default_models: Dict[str, str] = {}
    
    def register(self, spec: ModelSpec):
        """Register a model."""
        self.models[spec.model_id] = spec
    
    def get(self, model_id: str) -> Optional[ModelSpec]:
        """Get model by ID."""
        return self.models.get(model_id)
    
    def list_by_type(self, model_type: str) -> List[ModelSpec]:
        """List all models of a given type."""
        return [m for m in self.models.values() if m.model_type == model_type]
    
    def list_by_domain(self, domain: str) -> List[ModelSpec]:
        """List all models for a domain."""
        return [m for m in self.models.values() if m.domain == domain]
    
    def set_default(self, model_type: str, model_id: str):
        """Set default model for a type."""
        if model_id in self.models:
            self._default_models[model_type] = model_id
    
    def get_default(self, model_type: str) -> Optional[ModelSpec]:
        """Get default model for a type."""
        model_id = self._default_models.get(model_type)
        return self.get(model_id) if model_id else None
    
    def list_all(self) -> List[ModelSpec]:
        """List all registered models."""
        return list(self.models.values())
    
    def clear(self):
        """Clear all registered models."""
        self.models.clear()
        self._default_models.clear()


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry
