import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("kaelum.threshold_calibrator")


@dataclass
class ThresholdDecision:
    score: float
    threshold: float
    predicted_positive: bool
    actual_positive: bool
    task_type: str
    timestamp: float


class ThresholdCalibrator:
    def __init__(self, data_dir: str = ".kaelum/calibration"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_file = self.data_dir / "decisions.jsonl"
        self.thresholds_file = self.data_dir / "optimal_thresholds.json"
        self.decisions: Dict[str, List[ThresholdDecision]] = {}
        self.optimal_thresholds: Dict[str, float] = {}
        self._load_data()
    
    def record_decision(self, score: float, threshold: float, actual_result: bool, 
                       task_type: str, timestamp: float):
        if task_type not in self.decisions:
            self.decisions[task_type] = []
        
        decision = ThresholdDecision(
            score=score,
            threshold=threshold,
            predicted_positive=score > threshold,
            actual_positive=actual_result,
            task_type=task_type,
            timestamp=timestamp
        )
        self.decisions[task_type].append(decision)
        
        if len(self.decisions[task_type]) % 20 == 0:
            self._save_data()
            self._compute_optimal_thresholds(task_type)
    
    def get_optimal_threshold(self, task_type: str, default: float = 0.5) -> float:
        if task_type in self.optimal_thresholds:
            return self.optimal_thresholds[task_type]
        return default
    
    def _compute_optimal_thresholds(self, task_type: str):
        if task_type not in self.decisions or len(self.decisions[task_type]) < 10:
            return
        
        decisions = self.decisions[task_type]
        scores = [d.score for d in decisions]
        actuals = [d.actual_positive for d in decisions]
        
        best_f1 = 0.0
        best_threshold = 0.5
        
        for threshold in np.arange(0.2, 0.85, 0.05):
            tp = sum(1 for s, a in zip(scores, actuals) if s > threshold and a)
            fp = sum(1 for s, a in zip(scores, actuals) if s > threshold and not a)
            fn = sum(1 for s, a in zip(scores, actuals) if s <= threshold and a)
            
            if tp + fp == 0 or tp + fn == 0:
                continue
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall == 0:
                continue
            
            f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        if best_f1 > 0:
            self.optimal_thresholds[task_type] = best_threshold
            logger.info(f"Updated optimal threshold for {task_type}: {best_threshold:.3f} (F1: {best_f1:.3f})")
            self._save_thresholds()
    
    def get_performance_stats(self, task_type: str) -> Optional[Dict]:
        if task_type not in self.decisions or len(self.decisions[task_type]) < 5:
            return None
        
        decisions = self.decisions[task_type]
        
        tp = sum(1 for d in decisions if d.predicted_positive and d.actual_positive)
        fp = sum(1 for d in decisions if d.predicted_positive and not d.actual_positive)
        tn = sum(1 for d in decisions if not d.predicted_positive and not d.actual_positive)
        fn = sum(1 for d in decisions if not d.predicted_positive and d.actual_positive)
        
        total = len(decisions)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_decisions': total,
            'optimal_threshold': self.optimal_thresholds.get(task_type, 0.5)
        }
    
    def _save_data(self):
        with open(self.decisions_file, 'w') as f:
            for task_type, decisions in self.decisions.items():
                for decision in decisions:
                    f.write(json.dumps(asdict(decision)) + '\n')
    
    def _load_data(self):
        if not self.decisions_file.exists():
            return
        
        try:
            with open(self.decisions_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    decision = ThresholdDecision(**data)
                    if decision.task_type not in self.decisions:
                        self.decisions[decision.task_type] = []
                    self.decisions[decision.task_type].append(decision)
            
            for task_type in self.decisions.keys():
                self._compute_optimal_thresholds(task_type)
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
    
    def _save_thresholds(self):
        with open(self.thresholds_file, 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
    
    def _load_thresholds(self):
        if not self.thresholds_file.exists():
            return
        
        try:
            with open(self.thresholds_file, 'r') as f:
                self.optimal_thresholds = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load optimal thresholds: {e}")
