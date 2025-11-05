import re
from typing import Dict, List, Optional
from transformers import pipeline


class ConclusionDetector:
    def __init__(self):
        try:
            self.classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
        except:
            self.classifier = None
        
        self.conclusion_keywords = {
            'strong': ['therefore', 'thus', 'hence', 'consequently', 'it follows that', 'we can conclude'],
            'medium': ['so', 'in conclusion', 'to summarize', 'in summary', 'overall', 'finally'],
            'weak': ['ultimately', 'essentially', 'basically']
        }
        
        self.negation_patterns = [
            r'\b(not|no|don\'t|doesn\'t|cannot|can\'t|didn\'t|never)\b.*\b(therefore|thus|conclude|hence)\b',
            r'\b(therefore|thus|conclude|hence)\b.*\b(not|no|incorrect|wrong|false)\b'
        ]
    
    def detect(self, text: str, context: List[str]) -> Dict[str, any]:
        if not text or not text.strip():
            return {'is_conclusion': False, 'confidence': 0.0}
        
        text_lower = text.lower().strip()
        
        # 1. Check for negation context
        is_negated = any(re.search(pattern, text_lower) for pattern in self.negation_patterns)
        if is_negated:
            return {'is_conclusion': False, 'confidence': 0.0, 'reason': 'negated_conclusion'}
        
        # 2. Keyword signal (weight: 0.35)
        keyword_signal = 0.0
        matched_keywords = []
        
        for strength, keywords in self.conclusion_keywords.items():
            weight = {'strong': 0.35, 'medium': 0.25, 'weak': 0.15}[strength]
            for kw in keywords:
                if self._check_keyword_positive_context(text_lower, kw):
                    keyword_signal = max(keyword_signal, weight)
                    matched_keywords.append(kw)
                    break
        
        # 3. Structural signal (weight: 0.25)
        position_signal = 0.0
        if context:
            is_last_step = (text == context[-1])
            is_near_end = len(context) > 2 and text in context[-2:]
            
            if is_last_step:
                position_signal = 0.25
            elif is_near_end:
                position_signal = 0.15
        
        # 4. Semantic classification (weight: 0.40)
        semantic_signal = 0.0
        if self.classifier:
            try:
                hypothesis = "This text presents a final conclusion or result."
                result = self.classifier(f"{text}", hypothesis)
                
                if result and len(result) > 0:
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    if 'ENTAILMENT' in label.upper():
                        semantic_signal = score * 0.40
            except:
                pass
        else:
            # Fallback: check for conclusion patterns
            conclusion_patterns = [
                r'\b(answer|result|solution)\s*(is|are|:|=)',
                r'\b(final|ultimate|resulting)\s+\w+',
                r'^(therefore|thus|hence|so),?\s+[^,]{10,}'
            ]
            if any(re.search(pattern, text_lower) for pattern in conclusion_patterns):
                semantic_signal = 0.30
        
        total_confidence = keyword_signal + position_signal + semantic_signal
        total_confidence = min(total_confidence, 1.0)
        
        is_conclusion = total_confidence > 0.55
        
        return {
            'is_conclusion': is_conclusion,
            'confidence': total_confidence,
            'signals': {
                'keywords': keyword_signal,
                'position': position_signal,
                'semantic': semantic_signal
            },
            'matched_keywords': matched_keywords
        }
    
    def _check_keyword_positive_context(self, text: str, keyword: str) -> bool:
        if keyword not in text:
            return False
        
        kw_idx = text.find(keyword)
        
        window_start = max(0, kw_idx - 30)
        window_end = min(len(text), kw_idx + len(keyword) + 30)
        window = text[window_start:window_end]
        
        negation_words = ['not', "don't", "doesn't", "cannot", "can't", 'never', 'no']
        for neg in negation_words:
            if neg in window:
                neg_idx = window.find(neg)
                kw_relative = kw_idx - window_start
                
                if abs(neg_idx - kw_relative) < 20:
                    return False
        
        return True
