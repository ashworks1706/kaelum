from typing import Dict

class AdaptivePenalty:
    
    BASE_PENALTIES = {
        "math": 0.06,
        "code": 0.08,
        "logic": 0.06,
        "factual": 0.05,
        "creative": 0.04,
        "analysis": 0.05
    }
    
    @staticmethod
    def get_penalty(worker_type: str, query_complexity: float) -> float:
        base_penalty = AdaptivePenalty.BASE_PENALTIES.get(worker_type, 0.06)
        
        if query_complexity < 0.3:
            adjustment = 1.8
        elif query_complexity < 0.5:
            adjustment = 1.3
        elif query_complexity < 0.7:
            adjustment = 1.0
        elif query_complexity < 0.85:
            adjustment = 0.7
        else:
            adjustment = 0.5
        
        adaptive_penalty = base_penalty * adjustment
        
        return max(0.02, min(adaptive_penalty, 0.15))
    
    @staticmethod
    def compute_complexity(query: str, context: Dict = None) -> float:
        import re
        
        words = query.split()
        word_count = len(words)
        
        technical_terms = len(re.findall(
            r'\b(?:algorithm|optimize|implement|derive|prove|analyze|calculate|evaluate|demonstrate|construct)\b',
            query.lower()
        ))
        
        nested_structures = query.count('(') + query.count('[') + query.count('{')
        
        conjunctions = len(re.findall(
            r'\b(?:and|or|but|while|when|if|unless|although|because|since|therefore|however)\b',
            query.lower()
        ))
        
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        has_multi_part = any(marker in query.lower() for marker in ['first', 'second', 'then', 'next', 'finally', 'additionally'])
        
        complexity_score = (
            (word_count / 80.0) * 0.25 +
            (technical_terms / 4.0) * 0.30 +
            (nested_structures / 4.0) * 0.15 +
            (conjunctions / 4.0) * 0.15 +
            lexical_diversity * 0.10 +
            (0.05 if has_multi_part else 0.0)
        )
        
        return min(max(complexity_score, 0.0), 1.0)
