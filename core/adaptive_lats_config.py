from typing import Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class LATSConfig:
    max_depth: int
    num_simulations: int
    exploration_weight: float = 1.414
    use_cache: bool = True


class AdaptiveLATSConfig:
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.performance_history = defaultdict(lambda: deque(maxlen=history_size))
        
        self.base_configs = {
            'simple': LATSConfig(max_depth=3, num_simulations=5),
            'medium': LATSConfig(max_depth=5, num_simulations=10),
            'complex': LATSConfig(max_depth=8, num_simulations=20),
            'very_complex': LATSConfig(max_depth=10, num_simulations=30)
        }
        
        self.worker_adjustments = {
            'math': {'depth_multiplier': 1.2, 'sim_multiplier': 1.1},
            'logic': {'depth_multiplier': 1.3, 'sim_multiplier': 1.2},
            'code': {'depth_multiplier': 1.1, 'sim_multiplier': 1.0},
            'factual': {'depth_multiplier': 0.8, 'sim_multiplier': 0.7},
            'creative': {'depth_multiplier': 1.0, 'sim_multiplier': 1.3},
            'analysis': {'depth_multiplier': 1.1, 'sim_multiplier': 1.2}
        }
    
    def get_config(self, query_complexity: float, worker_type: str) -> LATSConfig:
        base_category = self._categorize_complexity(query_complexity)
        base_config = self.base_configs[base_category]
        
        adjustments = self.worker_adjustments.get(worker_type, {'depth_multiplier': 1.0, 'sim_multiplier': 1.0})
        
        adjusted_depth = int(base_config.max_depth * adjustments['depth_multiplier'])
        adjusted_sims = int(base_config.num_simulations * adjustments['sim_multiplier'])
        
        if worker_type in self.performance_history and len(self.performance_history[worker_type]) >= 10:
            history = list(self.performance_history[worker_type])
            
            high_quality = [h for h in history if h['quality'] > 0.8]
            if high_quality:
                avg_depth = sum(h['depth'] for h in high_quality) / len(high_quality)
                avg_sims = sum(h['sims'] for h in high_quality) / len(high_quality)
                
                adjusted_depth = int(0.7 * adjusted_depth + 0.3 * avg_depth)
                adjusted_sims = int(0.7 * adjusted_sims + 0.3 * avg_sims)
        
        adjusted_depth = max(2, min(adjusted_depth, 12))
        adjusted_sims = max(3, min(adjusted_sims, 40))
        
        return LATSConfig(
            max_depth=adjusted_depth,
            num_simulations=adjusted_sims,
            exploration_weight=base_config.exploration_weight,
            use_cache=base_config.use_cache
        )
    
    def _categorize_complexity(self, complexity: float) -> str:
        if complexity < 0.3:
            return 'simple'
        elif complexity < 0.55:
            return 'medium'
        elif complexity < 0.8:
            return 'complex'
        else:
            return 'very_complex'
    
    def record_performance(self, worker_type: str, complexity: float, 
                          depth_used: int, sims_used: int, quality_score: float):
        self.performance_history[worker_type].append({
            'complexity': complexity,
            'depth': depth_used,
            'sims': sims_used,
            'quality': quality_score
        })
    
    def get_statistics(self, worker_type: str) -> Dict:
        if worker_type not in self.performance_history:
            return {'count': 0}
        
        history = list(self.performance_history[worker_type])
        if not history:
            return {'count': 0}
        
        avg_depth = sum(h['depth'] for h in history) / len(history)
        avg_sims = sum(h['sims'] for h in history) / len(history)
        avg_quality = sum(h['quality'] for h in history) / len(history)
        
        return {
            'count': len(history),
            'avg_depth': avg_depth,
            'avg_simulations': avg_sims,
            'avg_quality': avg_quality
        }
