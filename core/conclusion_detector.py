import re
from typing import Dict, List, Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np


class ConclusionDetector:
    def __init__(self):
        try:
            self.classifier = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
        except:
            self.classifier = None
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.conclusion_exemplars = [
            "Therefore, we can conclude that the hypothesis is correct.",
            "In conclusion, the results demonstrate a clear pattern.",
            "Hence, the final answer is 42.",
            "Thus, we have proven the theorem.",
            "So the solution to the problem is x = 5.",
            "Consequently, this approach is the most effective."
        ]
        
        self.conclusion_embedding = self.encoder.encode(
            " ".join(self.conclusion_exemplars),
            convert_to_tensor=False
        )
    
    def detect(self, text: str, context: List[str]) -> Dict[str, any]:
        if not text or not text.strip():
            return {'is_conclusion': False, 'confidence': 0.0}
        
        text_lower = text.lower().strip()
        
        negation_patterns = [
            r'\b(not|no|don\'t|doesn\'t|cannot|can\'t|didn\'t|never)\s+(?:\w+\s+){0,3}(therefore|thus|conclude|hence)\b',
            r'\b(therefore|thus|conclude|hence)\s+(?:\w+\s+){0,3}(not|no|incorrect|wrong|false)\b'
        ]
        
        is_negated = any(re.search(pattern, text_lower) for pattern in negation_patterns)
        if is_negated:
            return {'is_conclusion': False, 'confidence': 0.0, 'reason': 'negated_conclusion'}
        
        semantic_signal = self._semantic_classification(text)
        structural_signal = self._structural_analysis(text, context)
        marker_signal = self._marker_analysis(text)
        
        total_confidence = (
            semantic_signal * 0.50 +
            structural_signal * 0.30 +
            marker_signal * 0.20
        )
        
        total_confidence = min(total_confidence, 1.0)
        is_conclusion = total_confidence > 0.55
        
        return {
            'is_conclusion': is_conclusion,
            'confidence': total_confidence,
            'signals': {
                'semantic': semantic_signal,
                'structural': structural_signal,
                'markers': marker_signal
            }
        }
    
    def _semantic_classification(self, text: str) -> float:
        text_embedding = self.encoder.encode(text, convert_to_tensor=False)
        
        similarity = np.dot(text_embedding, self.conclusion_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(self.conclusion_embedding) + 1e-9
        )
        
        if self.classifier:
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
                    
                    return float(0.6 * similarity + 0.4 * nli_score)
            except:
                pass
        
        return float(similarity)
    
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
