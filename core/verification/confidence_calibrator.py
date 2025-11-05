from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict, deque


class ConfidenceCalibrator:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.worker_history = defaultdict(lambda: deque(maxlen=history_size))
        self.calibration_curves = {}
        
        self.base_adjustments = {
            'code_present': 0.15,
            'syntax_valid': 0.12,
            'language_detected': 0.08,
            'task_simple': 0.05,
            'task_complex': -0.03,
            'has_specifics': 0.10,
            'good_coherence': 0.20,
            'good_diversity': 0.15,
            'adequate_length': 0.08
        }
    
    def record_outcome(self, worker_type: str, predicted_confidence: float, 
                       actual_passed: bool, task_features: Dict[str, bool]):
        self.worker_history[worker_type].append({
            'predicted': predicted_confidence,
            'actual': 1.0 if actual_passed else 0.0,
            'features': task_features
        })
        
        if len(self.worker_history[worker_type]) >= 20:
            self._update_calibration(worker_type)
    
    def _update_calibration(self, worker_type: str):
        history = list(self.worker_history[worker_type])
        if len(history) < 20:
            return
        
        predictions = np.array([h['predicted'] for h in history])
        actuals = np.array([h['actual'] for h in history])
        
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(predictions, bins) - 1
        
        calibration = {}
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_center = (bins[i] + bins[i+1]) / 2
                actual_accuracy = np.mean(actuals[mask])
                calibration[bin_center] = actual_accuracy
        
        self.calibration_curves[worker_type] = calibration
    
    def calibrate_confidence(self, worker_type: str, raw_confidence: float,
                            task_features: Dict[str, bool]) -> float:
        if worker_type in self.calibration_curves and self.calibration_curves[worker_type]:
            calibrated = self._apply_calibration_curve(worker_type, raw_confidence)
        else:
            calibrated = raw_confidence
        
        adjustment = 0.0
        for feature, is_present in task_features.items():
            if is_present and feature in self.base_adjustments:
                base_adj = self.base_adjustments[feature]
                
                if worker_type in self.worker_history and len(self.worker_history[worker_type]) >= 10:
                    feature_performance = self._compute_feature_performance(worker_type, feature)
                    adjustment += base_adj * feature_performance
                else:
                    adjustment += base_adj
        
        final_confidence = calibrated + adjustment
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _apply_calibration_curve(self, worker_type: str, confidence: float) -> float:
        curve = self.calibration_curves[worker_type]
        bin_centers = sorted(curve.keys())
        
        if confidence <= bin_centers[0]:
            return curve[bin_centers[0]]
        if confidence >= bin_centers[-1]:
            return curve[bin_centers[-1]]
        
        for i in range(len(bin_centers) - 1):
            if bin_centers[i] <= confidence <= bin_centers[i+1]:
                x0, y0 = bin_centers[i], curve[bin_centers[i]]
                x1, y1 = bin_centers[i+1], curve[bin_centers[i+1]]
                alpha = (confidence - x0) / (x1 - x0)
                return y0 + alpha * (y1 - y0)
        
        return confidence
    
    def _compute_feature_performance(self, worker_type: str, feature: str) -> float:
        history = list(self.worker_history[worker_type])
        
        with_feature = [h for h in history if h['features'].get(feature, False)]
        without_feature = [h for h in history if not h['features'].get(feature, False)]
        
        if len(with_feature) < 3 or len(without_feature) < 3:
            return 1.0
        
        acc_with = np.mean([h['actual'] for h in with_feature])
        acc_without = np.mean([h['actual'] for h in without_feature])
        
        if acc_without == 0:
            return 1.5
        
        relative_improvement = acc_with / acc_without
        return np.clip(relative_improvement, 0.5, 2.0)
    
    def get_statistics(self, worker_type: str) -> Dict:
        if worker_type not in self.worker_history:
            return {'count': 0}
        
        history = list(self.worker_history[worker_type])
        if not history:
            return {'count': 0}
        
        predictions = np.array([h['predicted'] for h in history])
        actuals = np.array([h['actual'] for h in history])
        
        mae = np.mean(np.abs(predictions - actuals))
        accuracy = np.mean(actuals)
        
        return {
            'count': len(history),
            'mean_absolute_error': float(mae),
            'accuracy': float(accuracy),
            'is_calibrated': worker_type in self.calibration_curves
        }
