import math
from typing import Dict, Any, Optional

_feedback_engine = None
_prm = None

def get_feedback_engine():
    """Lazy load feedback engine to avoid circular imports."""
    global _feedback_engine
    if _feedback_engine is None:
        from ..learning.human_feedback import HumanFeedbackEngine
        _feedback_engine = HumanFeedbackEngine()
    return _feedback_engine


def _get_prm():
    """Lazy load the learned PRM singleton."""
    global _prm
    if _prm is None:
        from ..verification.process_reward_model import get_prm
        _prm = get_prm()
    return _prm

class RewardModel:
    @staticmethod
    def compute_confidence(node_value: float, node_visits: int, total_visits: int) -> float:
        if node_visits == 0:
            return 0.3
        
        answer_quality = node_value / node_visits
        exploration_confidence = math.sqrt(node_visits / max(total_visits, 1))
        
        confidence = (exploration_confidence * 0.3 + answer_quality * 0.7)
        return min(max(confidence, 0.0), 1.0)
    
    @staticmethod
    def get_reward(worker_type: str, state: Dict[str, Any], depth: int,
                   has_answer: bool = False, has_partial: bool = False,
                   syntax_valid: bool = False, query_complexity: float = 0.5) -> float:
        import logging
        logger = logging.getLogger("kaelum.reward")
        
        prm = _get_prm()
        if not prm or not prm.is_active:
            raise RuntimeError("PRM is not active; cannot compute reward without trained PRM.")

        step_text = state.get("step", "")
        query_text = state.get("query", "")
        if not step_text or not query_text:
            raise RuntimeError("Missing step or query text for PRM scoring.")

        prm_score = prm.predict_step_quality(
            query=query_text,
            step=step_text,
            worker_type=worker_type,
        )
        if prm_score is None:
            raise RuntimeError("PRM did not return a score; aborting reward computation.")
        base_reward = prm_score
        logger.debug(f"REWARD [{worker_type}]: PRM score {prm_score:.3f}")

        # ── Human feedback adjustment ─────────────────────────────────────────
        feedback_engine = get_feedback_engine()
        if feedback_engine:
            adjusted_reward = feedback_engine.get_adjusted_reward(
                worker_type=worker_type,
                base_reward=base_reward,
                is_partial=has_partial
            )
            
            if abs(adjusted_reward - base_reward) > 0.001:
                logger.debug(f"REWARD [{worker_type}]: Human feedback adjustment: {base_reward:.3f} → {adjusted_reward:.3f}")
            
            return adjusted_reward
        
        return base_reward
