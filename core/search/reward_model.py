import math
from typing import Dict, Any, Optional
from ..learning import AdaptivePenalty

# Late import to avoid circular dependency
_feedback_engine = None

def get_feedback_engine():
    """Lazy load feedback engine to avoid circular imports."""
    global _feedback_engine
    if _feedback_engine is None:
        try:
            from ..learning.human_feedback import HumanFeedbackEngine
            _feedback_engine = HumanFeedbackEngine()
        except:
            _feedback_engine = None
    return _feedback_engine


class RewardModel:
    CONFIGS = {
        "math": {
            "has_answer": 0.85,
            "partial": 0.50,
            "base": 0.40,
            "depth_penalty": 0.06
        },
        "code": {
            "syntax_valid": 0.90,
            "has_code": 0.80,
            "partial": 0.60,
            "base": 0.30,
            "depth_penalty": 0.08
        },
        "logic": {
            "has_conclusion": 0.88,
            "partial": 0.58,
            "base": 0.42,
            "depth_penalty": 0.06
        },
        "factual": {
            "has_facts": 0.82,
            "partial": 0.55,
            "base": 0.38,
            "depth_penalty": 0.05
        },
        "creative": {
            "complete": 0.80,
            "partial": 0.55,
            "base": 0.35,
            "depth_penalty": 0.04
        },
        "analysis": {
            "complete": 0.83,
            "partial": 0.57,
            "base": 0.40,
            "depth_penalty": 0.05
        }
    }
    
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
        
        config = RewardModel.CONFIGS.get(worker_type, RewardModel.CONFIGS["math"])
        
        # Get base reward from config
        base_reward = 0.0
        
        if has_answer:
            if worker_type == "code" and syntax_valid:
                base_reward = config["syntax_valid"]
                logger.debug(f"REWARD [{worker_type}]: {base_reward:.3f} (syntax valid code at depth {depth})")
            else:
                base_reward = config.get("has_answer", config.get("complete", 0.85))
                logger.debug(f"REWARD [{worker_type}]: {base_reward:.3f} (complete answer at depth {depth})")
        
        elif has_partial:
            base_reward = config["partial"]
            logger.debug(f"REWARD [{worker_type}]: {base_reward:.3f} (partial solution at depth {depth})")
        
        else:
            adaptive_penalty = AdaptivePenalty.get_penalty(worker_type, query_complexity)
            base_reward = max(0.1, config["base"] - depth * adaptive_penalty)
            logger.debug(f"REWARD [{worker_type}]: {base_reward:.3f} (base={config['base']:.2f}, depth_penalty={adaptive_penalty:.3f}, depth={depth})")
        
        # Apply human feedback adjustments
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
