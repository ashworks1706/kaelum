"""Model registry for domain-specific and specialized models."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    model_id: str
    model_type: str  # reasoning, planning, routing, vision, math
    base_url: str
    description: str
    domain: Optional[str] = None  # e.g., "medical", "legal", "code", "math"
    context_length: int = 4096
    quantization: Optional[str] = None  # e.g., "4bit", "8bit"
    vram_gb: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MathCapabilities:
    """Math-specific capabilities for a model."""
    symbolic_computation: bool = False
    calculus: bool = False
    equation_solving: bool = False
    verification: bool = False
    multivariate: bool = False
    strict_formatting: bool = False


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self._default_models: Dict[str, str] = {}
        self._math_capabilities: Dict[str, MathCapabilities] = {}
    
    def register(self, spec: ModelSpec, math_capabilities: Optional[MathCapabilities] = None):
        """Register a model with optional math capabilities."""
        self.models[spec.model_id] = spec
        if math_capabilities:
            self._math_capabilities[spec.model_id] = math_capabilities
    
    def register_math_model(self, spec: ModelSpec, capabilities: MathCapabilities):
        """Register a model specifically for math tasks."""
        if spec.domain != "math":
            spec.domain = "math"
        self.register(spec, capabilities)
    
    def get(self, model_id: str) -> Optional[ModelSpec]:
        """Get model by ID."""
        return self.models.get(model_id)
    
    def get_math_capabilities(self, model_id: str) -> Optional[MathCapabilities]:
        """Get math capabilities for a model."""
        return self._math_capabilities.get(model_id)
    
    def list_by_type(self, model_type: str) -> List[ModelSpec]:
        """List all models of a given type."""
        return [m for m in self.models.values() if m.model_type == model_type]
    
    def list_by_domain(self, domain: str) -> List[ModelSpec]:
        """List all models for a domain."""
        return [m for m in self.models.values() if m.domain == domain]
    
    def list_math_models(self) -> List[ModelSpec]:
        """List all models with math capabilities."""
        return [m for m in self.models.values() if m.model_id in self._math_capabilities]
    
    def find_best_math_model(self, required_capabilities: List[str]) -> Optional[ModelSpec]:
        """Find the best model for specific math capabilities."""
        for model_id, capabilities in self._math_capabilities.items():
            if all(getattr(capabilities, cap, False) for cap in required_capabilities):
                return self.models.get(model_id)
        return None
    
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
        self._math_capabilities.clear()


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry


def register_default_math_models():
    """Register some default models with math capabilities."""
    registry = get_registry()
    
    # Register Llama 3.1 8B with strong math capabilities
    llama_math = ModelSpec(
        model_id="llama3.1:8b",
        model_type="reasoning",
        base_url="http://localhost:11434/v1",
        description="Llama 3.1 8B with strong mathematical reasoning capabilities",
        domain="math",
        context_length=8192,
        vram_gb=6.0,
        metadata={"supports_function_calling": True}
    )
    
    llama_capabilities = MathCapabilities(
        symbolic_computation=True,
        calculus=True, 
        equation_solving=True,
        verification=True,
        multivariate=True,
        strict_formatting=False
    )
    
    registry.register_math_model(llama_math, llama_capabilities)
    
    # Register Qwen models for math
    qwen_math = ModelSpec(
        model_id="qwen3:4b",
        model_type="reasoning", 
        base_url="http://localhost:11434/v1",
        description="Qwen 3 4B optimized for mathematical reasoning",
        domain="math",
        context_length=4096,
        vram_gb=3.0
    )
    
    qwen_capabilities = MathCapabilities(
        symbolic_computation=True,
        calculus=True,
        equation_solving=True,
        verification=True,
        multivariate=False,
        strict_formatting=True
    )
    
    registry.register_math_model(qwen_math, qwen_capabilities)
    
    # Register math-specialized coder model
    qwen_coder = ModelSpec(
        model_id="qwen3-coder:30b",
        model_type="reasoning",
        base_url="http://localhost:11434/v1", 
        description="Qwen 3 Coder 30B with advanced mathematical and coding capabilities",
        domain="math",
        context_length=8192,
        vram_gb=18.0
    )
    
    coder_capabilities = MathCapabilities(
        symbolic_computation=True,
        calculus=True,
        equation_solving=True,
        verification=True,
        multivariate=True,
        strict_formatting=True
    )
    
    registry.register_math_model(qwen_coder, coder_capabilities)
    
    # Set defaults
    registry.set_default("reasoning", "llama3.1:8b")
    registry.set_default("math", "llama3.1:8b")
