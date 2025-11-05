import math
from typing import Dict, Any
from ..learning import AdaptivePenalty


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
        config = RewardModel.CONFIGS.get(worker_type, RewardModel.CONFIGS["math"])
        
        if has_answer:
            if worker_type == "code" and syntax_valid:
                return config["syntax_valid"]
            return config.get("has_answer", config.get("complete", 0.85))
        
        if has_partial:
            return config["partial"]
        
        adaptive_penalty = AdaptivePenalty.get_penalty(worker_type, query_complexity)
        
        return max(0.1, config["base"] - depth * adaptive_penalty)
