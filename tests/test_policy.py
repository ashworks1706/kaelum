"""Tests for policy module."""

import pytest
from kaelum.core.policy import PolicyController, LatencyOptimizer


class TestPolicyController:
    """Tests for PolicyController."""

    def test_initialization(self):
        """Test policy controller initialization."""
        controller = PolicyController(enable_learning=True)

        assert controller.enable_learning is True
        assert "min_confidence_threshold" in controller.policy
        assert "max_reflection_iterations" in controller.policy
        assert controller.state["total_queries"] == 0

    def test_get_policy_simple_query(self):
        """Test policy for simple query."""
        controller = PolicyController(enable_learning=True)

        policy = controller.get_policy_for_query("What is 2 + 2?")

        assert "max_reflection_iterations" in policy
        # Simple query should have fewer iterations
        assert policy["max_reflection_iterations"] <= 2

    def test_get_policy_complex_query(self):
        """Test policy for complex query."""
        controller = PolicyController(enable_learning=True)

        long_query = " ".join(["word"] * 60)  # 60 word query
        policy = controller.get_policy_for_query(long_query)

        # Complex query should have more iterations
        assert policy["max_reflection_iterations"] >= 2

    def test_update_policy(self):
        """Test policy update based on results."""
        controller = PolicyController(enable_learning=True)

        # Record a successful result
        result = {"verified": True, "confidence": 0.9}
        controller.update_policy(result)

        assert controller.state["total_queries"] == 1
        assert controller.state["success_count"] == 1
        assert controller.state["avg_confidence"] > 0

    def test_policy_adaptation(self):
        """Test that policy adapts to poor performance."""
        controller = PolicyController(enable_learning=True)

        # Simulate multiple poor results
        for _ in range(15):
            result = {"verified": False, "confidence": 0.4}
            controller.update_policy(result)

        # Policy should become stricter
        initial_threshold = 0.7
        assert controller.policy["min_confidence_threshold"] >= initial_threshold

    def test_no_learning_mode(self):
        """Test policy controller with learning disabled."""
        controller = PolicyController(enable_learning=False)

        policy = controller.get_policy_for_query("Test query")
        initial_policy = controller.policy.copy()

        # Update with result
        result = {"verified": True, "confidence": 0.9}
        controller.update_policy(result)

        # Policy should not change
        assert controller.policy == initial_policy

    def test_get_state(self):
        """Test getting policy state."""
        controller = PolicyController(enable_learning=True)

        state = controller.get_state()

        assert "policy" in state
        assert "state" in state
        assert "learning_enabled" in state
        assert state["learning_enabled"] is True

    def test_reset(self):
        """Test resetting policy controller."""
        controller = PolicyController(enable_learning=True)

        # Make some updates
        result = {"verified": True, "confidence": 0.9}
        controller.update_policy(result)
        controller.update_policy(result)

        # Reset
        controller.reset()

        assert controller.state["total_queries"] == 0
        assert controller.state["success_count"] == 0


class TestLatencyOptimizer:
    """Tests for LatencyOptimizer."""

    def test_should_skip_verification_low_budget(self):
        """Test skip decision with low time budget."""
        optimizer = LatencyOptimizer()

        # Low complexity, low budget -> skip
        should_skip = optimizer.should_skip_verification(
            query_complexity=0.2, time_budget=0.5
        )
        assert should_skip is True

    def test_should_skip_verification_high_complexity(self):
        """Test skip decision with high complexity."""
        optimizer = LatencyOptimizer()

        # High complexity, low budget -> don't skip
        should_skip = optimizer.should_skip_verification(
            query_complexity=0.8, time_budget=0.5
        )
        assert should_skip is False

    def test_should_skip_verification_high_budget(self):
        """Test skip decision with high time budget."""
        optimizer = LatencyOptimizer()

        # Any complexity, high budget -> don't skip
        should_skip = optimizer.should_skip_verification(
            query_complexity=0.2, time_budget=5.0
        )
        assert should_skip is False

    def test_get_optimal_iterations_low_budget(self):
        """Test iteration count with low budget."""
        optimizer = LatencyOptimizer()

        iterations = optimizer.get_optimal_iterations(time_budget=2.0)
        assert iterations == 1

    def test_get_optimal_iterations_medium_budget(self):
        """Test iteration count with medium budget."""
        optimizer = LatencyOptimizer()

        iterations = optimizer.get_optimal_iterations(time_budget=5.0)
        assert iterations == 2

    def test_get_optimal_iterations_high_budget(self):
        """Test iteration count with high budget."""
        optimizer = LatencyOptimizer()

        iterations = optimizer.get_optimal_iterations(time_budget=10.0)
        assert iterations == 3
