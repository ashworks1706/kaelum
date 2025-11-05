import re
from typing import Dict, List, Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from ..verification.threshold_calibrator import ThresholdCalibrator


class ConclusionDetector:
    def __init__(self):
        try:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
            self.use_zero_shot = True
        except:
            try:
                self.classifier = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
                self.use_zero_shot = False
            except:
                self.classifier = None
                self.use_zero_shot = False
        
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.threshold_calibrator = ThresholdCalibrator()
        
        self.conclusion_exemplars = [
            "Therefore, we can conclude that the hypothesis is correct.",
            "In conclusion, the results demonstrate a clear pattern.",
            "Hence, the final answer is 42.",
            "Thus, we have proven the theorem.",
            "So the solution to the problem is x = 5.",
            "Consequently, this approach is the most effective.",
            "This proves that the statement is true.",
            "The answer is therefore 10.",
            "We conclude that this is the optimal solution."
        ]
        
        self.non_conclusion_exemplars = [
            "We need to explore various possibilities.",
            "Let's consider the following approach.",
            "The data shows interesting patterns.",
            "First, we will analyze the input.",
            "This requires further investigation."
        ]
        
        self.conclusion_embedding = self.encoder.encode(
            self.conclusion_exemplars,
            convert_to_tensor=False
        ).mean(axis=0)
        
        self.non_conclusion_embedding = self.encoder.encode(
            self.non_conclusion_exemplars,
            convert_to_tensor=False
        ).mean(axis=0)
    
    def detect(self, text: str, context: List[str]) -> Dict[str, any]:
        if not text or not text.strip():
            return {'is_conclusion': False, 'confidence': 0.0}
        
        text_lower = text.lower().strip()
        
        if self._contains_strong_negation(text_lower):
            return {'is_conclusion': False, 'confidence': 0.0, 'reason': 'negated_conclusion'}
        
        semantic_signal = self._semantic_classification(text)
        structural_signal = self._structural_analysis(text, context)
        marker_signal = self._marker_analysis_contextual(text)
        
        total_confidence = (
            semantic_signal * 0.50 +
            structural_signal * 0.30 +
            marker_signal * 0.20
        )
        
        total_confidence = min(total_confidence, 1.0)
        
        threshold = self._adaptive_threshold(text, context)
        is_conclusion = total_confidence > threshold
        
        return {
            'is_conclusion': is_conclusion,
            'confidence': total_confidence,
            'signals': {
                'semantic': semantic_signal,
                'structural': structural_signal,
                'markers': marker_signal
            }
        }
    
    def _adaptive_threshold(self, text: str, context: List[str]) -> float:
        optimal_threshold = self.threshold_calibrator.get_optimal_threshold(
            "conclusion_detection",
            default=0.55
        )
        
        words = text.split()
        if len(words) < 10:
            optimal_threshold += 0.1
        elif len(words) > 30:
            optimal_threshold -= 0.05
        
        if context and text == context[-1]:
            optimal_threshold -= 0.05
        
        return max(0.45, min(0.70, optimal_threshold))
    
    def _contains_strong_negation(self, text: str) -> bool:
        negation_contexts = [
            r'\b(not|no|don\'t|doesn\'t|cannot|can\'t|didn\'t|never)\s+(?:\w+\s+){0,3}(therefore|thus|conclude|hence|answer|result|solution)\b',
            r'\b(therefore|thus|conclude|hence|answer|result|solution)\s+(?:\w+\s+){0,3}(not|no|incorrect|wrong|false|invalid)\b',
            r'\b(cannot|can\'t)\s+conclude\b',
            r'\bno\s+(conclusion|answer|solution)\b'
        ]
        
        return any(re.search(pattern, text) for pattern in negation_contexts)
    
    def _semantic_classification(self, text: str) -> float:
        text_embedding = self.encoder.encode(text, convert_to_tensor=False)
        
        conclusion_sim = np.dot(text_embedding, self.conclusion_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(self.conclusion_embedding) + 1e-9
        )
        
        non_conclusion_sim = np.dot(text_embedding, self.non_conclusion_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(self.non_conclusion_embedding) + 1e-9
        )
        
        contrastive_score = (conclusion_sim - non_conclusion_sim + 1.0) / 2.0
        
        if self.classifier and self.use_zero_shot:
            try:
                result = self.classifier(text, candidate_labels=["conclusion", "explanation", "question"], hypothesis_template="This text is a {}.")
                
                conclusion_idx = result['labels'].index('conclusion')
                zero_shot_score = result['scores'][conclusion_idx]
                
                return float(0.5 * contrastive_score + 0.5 * zero_shot_score)
            except:
                pass
        elif self.classifier:
            try:
                hypothesis = "This text presents a final conclusion or result."
                result = self.classifier(f"{text}", hypothesis)
                
                if result and len(result) > 0:
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    if 'ENTAILMENT' in label.upper():
                        nli_score = score
                    elif 'NEUTRAL' in label.upper():
                        nli_score = score * 0.5
                    else:
                        nli_score = 0.0
                    
                    return float(0.5 * contrastive_score + 0.5 * nli_score)
            except:
                pass
        
        return float(contrastive_score)
    
    def _structural_analysis(self, text: str, context: List[str]) -> float:
        if not context:
            return 0.3
        
        position_signal = 0.0
        if text == context[-1]:
            position_signal = 0.8
        elif len(context) > 2 and text in context[-2:]:
            position_signal = 0.5
        elif len(context) > 3 and text in context[-3:]:
            position_signal = 0.3
        
        conclusion_patterns = [
            r'\b(answer|result|solution|outcome)\s*(is|are|:|=|equals)\b',
            r'\b(final|ultimate|resulting|conclusive)\s+\w+',
            r'^(therefore|thus|hence|so|consequently),?\s+[^,]{10,}'
        ]
        
        pattern_match = any(re.search(pattern, text.lower()) for pattern in conclusion_patterns)
        pattern_signal = 0.4 if pattern_match else 0.0
        
        return min(0.6 * position_signal + 0.4 * pattern_signal, 1.0)
    
    def _marker_analysis_contextual(self, text: str) -> float:
        text_lower = text.lower()
        
        marker_configs = {
            'strong': {
                'markers': ['therefore', 'thus', 'hence', 'consequently', 'it follows that', 'we can conclude'],
                'base_weight': 0.30
            },
            'medium': {
                'markers': ['so', 'in conclusion', 'to summarize', 'in summary', 'overall', 'finally'],
                'base_weight': 0.20
            },
            'weak': {
                'markers': ['ultimately', 'essentially', 'basically'],
                'base_weight': 0.10
            }
        }
        
        max_score = 0.0
        
        for strength, config in marker_configs.items():
            for marker in config['markers']:
                if marker in text_lower:
                    context_score = self._analyze_marker_context(text_lower, marker)
                    adjusted_weight = config['base_weight'] * context_score
                    max_score = max(max_score, adjusted_weight)
        
        return max_score
    
    def _analyze_marker_context(self, text: str, marker: str) -> float:
        if marker not in text:
            return 0.0
        
        marker_idx = text.find(marker)
        
        window_start = max(0, marker_idx - 40)
        window_end = min(len(text), marker_idx + len(marker) + 40)
        window = text[window_start:window_end]
        
        negation_patterns = [
            r'\b(not|no|don\'t|doesn\'t|cannot|can\'t|never|won\'t|wouldn\'t)\b',
            r'\b(but|however|although|though|unless|except)\b'
        ]
        
        for pattern in negation_patterns:
            matches = list(re.finditer(pattern, window))
            for match in matches:
                neg_pos = match.start()
                marker_relative = marker_idx - window_start
                
                if abs(neg_pos - marker_relative) < 25:
                    return 0.3
        
        confirmation_patterns = [
            r'\b(answer|result|solution|proof|finding)\s+(is|are)\b',
            r'\b(final|ultimate|definitive)\b',
            r'[=:]'
        ]
        
        has_confirmation = any(re.search(p, window) for p in confirmation_patterns)
        
        return 1.0 if has_confirmation else 0.7
    
    def _marker_analysis(self, text: str) -> float:
        text_lower = text.lower()
        
        strong_markers = {
            'therefore': 0.30,
            'thus': 0.30,
            'hence': 0.30,
            'consequently': 0.28,
            'it follows that': 0.28,
            'we can conclude': 0.32
        }
        
        medium_markers = {
            'so': 0.15,
            'in conclusion': 0.25,
            'to summarize': 0.22,
            'in summary': 0.22,
            'overall': 0.18,
            'finally': 0.20
        }
        
        weak_markers = {
            'ultimately': 0.10,
            'essentially': 0.08,
            'basically': 0.08
        }
        
        max_score = 0.0
        
        for marker, score in strong_markers.items():
            if self._check_marker_context(text_lower, marker):
                max_score = max(max_score, score)
        
        if max_score < 0.20:
            for marker, score in medium_markers.items():
                if self._check_marker_context(text_lower, marker):
                    max_score = max(max_score, score)
        
        if max_score < 0.10:
            for marker, score in weak_markers.items():
                if self._check_marker_context(text_lower, marker):
                    max_score = max(max_score, score)
        
        return max_score
    
    def _check_marker_context(self, text: str, marker: str) -> bool:
        if marker not in text:
            return False
        
        marker_idx = text.find(marker)
        
        window_start = max(0, marker_idx - 30)
        window_end = min(len(text), marker_idx + len(marker) + 30)
        window = text[window_start:window_end]
        
        negation_words = ['not', "don't", "doesn't", "cannot", "can't", 'never', 'no', "won't", "wouldn't"]
        for neg in negation_words:
            if neg in window:
                neg_idx = window.find(neg)
                marker_relative = marker_idx - window_start
                
                if abs(neg_idx - marker_relative) < 20:
                    return False
        
        return True
    
    def record_outcome(self, score: float, threshold: float, was_correct: bool):
        self.threshold_calibrator.record_decision(
            score=score,
            threshold=threshold,
            actual_result=was_correct,
            task_type="conclusion_detection",
            timestamp=time.time()
        )
