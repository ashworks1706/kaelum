"""Tests for metrics and cost tracking."""

import pytest
import time
from kaelum.core.metrics import CostTracker


class TestCostTracker:
    """Test cost tracking functionality."""
    
    def test_session_creation(self):
        """Test creating tracking sessions."""
        tracker = CostTracker()
        
        session_id = "test-session-1"
        tracker.start_session(session_id, metadata={"test": "data"})
        
        assert session_id in tracker.sessions
        assert tracker.sessions[session_id]["metadata"]["test"] == "data"
    
    def test_log_inference(self):
        """Test logging inference calls."""
        tracker = CostTracker()
        session_id = "test-session"
        
        tracker.start_session(session_id)
        tracker.log_inference(
            model_type="local_reasoning",
            tokens=100,
            latency_ms=150.5,
            cost=0.000001,
            session_id=session_id
        )
        
        metrics = tracker.get_session_metrics(session_id)
        
        assert metrics["total_tokens"] == 100
        assert metrics["total_cost"] == 0.000001
        assert metrics["total_inferences"] == 1
    
    def test_multiple_inferences(self):
        """Test tracking multiple inferences."""
        tracker = CostTracker()
        session_id = "test-session"
        
        tracker.start_session(session_id)
        
        # Log multiple calls
        tracker.log_inference("local_reasoning", 100, 150, 0.000001, session_id)
        tracker.log_inference("local_reasoning", 200, 200, 0.000002, session_id)
        tracker.log_inference("local_answer", 50, 80, 0.0000005, session_id)
        
        metrics = tracker.get_session_metrics(session_id)
        
        assert metrics["total_tokens"] == 350
        assert metrics["total_cost"] == pytest.approx(0.0000035, rel=1e-9)
        assert metrics["total_inferences"] == 3
    
    def test_calculate_savings(self):
        """Test cost savings calculation."""
        tracker = CostTracker()
        session_id = "test-session"
        
        tracker.start_session(session_id)
        tracker.log_inference("local_reasoning", 1000, 200, 0.00001, session_id)
        
        savings = tracker.calculate_savings(session_id)
        
        assert "actual_cost" in savings
        assert "commercial_cost" in savings
        assert "savings" in savings
        assert "savings_percent" in savings
        assert savings["savings"] > 0
        assert savings["savings_percent"] > 0
    
    def test_export_session(self):
        """Test exporting session data."""
        tracker = CostTracker()
        session_id = "test-session"
        
        tracker.start_session(session_id, metadata={"query": "test"})
        tracker.log_inference("local_reasoning", 100, 150, 0.000001, session_id)
        
        json_data = tracker.export_session(session_id)
        
        assert json_data is not None
        assert "start_time" in json_data
        assert "metadata" in json_data
        assert "query" in json_data
    
    def test_nonexistent_session(self):
        """Test handling of nonexistent sessions."""
        tracker = CostTracker()
        
        # Should handle gracefully
        metrics = tracker.get_session_metrics("nonexistent")
        assert metrics == {}


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_import(self):
        """Test importing registry."""
        from kaelum.core.registry import ModelRegistry, ModelSpec, get_registry
        
        registry = get_registry()
        assert isinstance(registry, ModelRegistry)
    
    def test_register_model(self):
        """Test registering a model."""
        from kaelum.core.registry import ModelRegistry, ModelSpec
        
        registry = ModelRegistry()
        
        spec = ModelSpec(
            model_id="test-model",
            model_type="reasoning",
            base_url="http://localhost:8000/v1",
            description="A test model",
            domain="general"
        )
        
        registry.register(spec)
        
        retrieved = registry.get("test-model")
        assert retrieved is not None
        assert retrieved.model_id == "test-model"
        assert retrieved.model_type == "reasoning"
    
    def test_list_by_type(self):
        """Test listing models by type."""
        from kaelum.core.registry import ModelRegistry, ModelSpec
        
        registry = ModelRegistry()
        registry.clear()
        
        spec1 = ModelSpec(
            model_id="model-1",
            model_type="reasoning",
            base_url="http://localhost:8000/v1",
            description="Model 1"
        )
        spec2 = ModelSpec(
            model_id="model-2",
            model_type="planning",
            base_url="http://localhost:8000/v1",
            description="Model 2"
        )
        
        registry.register(spec1)
        registry.register(spec2)
        
        reasoning_models = registry.list_by_type("reasoning")
        assert len(reasoning_models) == 1
        assert reasoning_models[0].model_id == "model-1"
    
    def test_default_model(self):
        """Test setting and getting default models."""
        from kaelum.core.registry import ModelRegistry, ModelSpec
        
        registry = ModelRegistry()
        registry.clear()
        
        spec = ModelSpec(
            model_id="default-model",
            model_type="reasoning",
            base_url="http://localhost:8000/v1",
            description="Default Model"
        )
        
        registry.register(spec)
        registry.set_default("reasoning", "default-model")
        
        default = registry.get_default("reasoning")
        assert default is not None
        assert default.model_id == "default-model"
