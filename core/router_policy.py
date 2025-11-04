"""Router policy interface for Kaelum routing-first framework.

This file defines a simple, minimal interface for pluggable routing policies.
Implementations may be supervised models, contextual bandits, or RL agents.

Keep this file tiny — it is the contract between the orchestrator (environment)
and any routing policy you want to experiment with.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RouterPolicy:
    """Abstract routing policy interface.

    Methods are intentionally small and synchronous to make testing and
    iteration easy. Implementations may be pure in-memory (for research)
    or backed by PyTorch/TensorFlow models for inference.
    """

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return a routing decision for the provided features.

        Args:
            features: a serializable dict describing the query and context.

        Returns:
            action: dict describing routing action. Example keys:
                - "expert_id": str
                - "use_symbolic_verification": bool
                - "max_reflection_iterations": int
                - "confidence_threshold": float
        """
        raise NotImplementedError

    def record(self, state: Dict[str, Any], action: Dict[str, Any],
               reward: float, next_state: Optional[Dict[str, Any]], done: bool) -> None:
        """Record a transition for offline/online training or logging.

        Implementations may append to a replay buffer, write to disk, or
        no-op (if logging is handled elsewhere).
        """
        raise NotImplementedError

    def train_step(self) -> Dict[str, float]:
        """Perform an optional training/update step and return metrics.

        Returns a small dict of training metrics (loss, lr, steps, etc.).
        This method is intentionally optional for policies that don't train
        in-process.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save policy state (weights, metadata) to path."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load policy state from path."""
        raise NotImplementedError


class BaselineRulePolicy(RouterPolicy):
    """Tiny rule-based baseline that picks a default expert.

    Useful as a fallback during development. Keep implementation minimal.
    """

    def __init__(self, default_expert: str = "expert_general"):
        self.default_expert = default_expert
        # lightweight in-memory buffer for recorded transitions (optional)
        self._buffer = []

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "expert_id": self.default_expert,
            "use_symbolic_verification": True,
            "max_reflection_iterations": 1,
            "confidence_threshold": 0.75,
            "policy": "baseline_rule",
        }

    def record(self, state: Dict[str, Any], action: Dict[str, Any],
               reward: float, next_state: Optional[Dict[str, Any]], done: bool) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> Dict[str, float]:
        # No-op baseline; return empty metrics
        return {"updated": 0}

    def save(self, path: str) -> None:
        # small helper for dev: write buffer length
        try:
            with open(path, "w") as f:
                f.write(str(len(self._buffer)))
        except Exception:
            pass

    def load(self, path: str) -> None:
        # no model to load for rule-based baseline
        return
