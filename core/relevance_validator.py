import re
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer, util


class RelevanceValidator:
    
    BASE_THRESHOLDS = {
        'code_worker': 0.40,
        'factual_worker': 0.50,
        'creative_worker': 0.35,
        'math_worker': 0.45,
        'logic_worker': 0.45,
        'analysis_worker': 0.42,
        'default': 0.40
    }
    
    COMPLEXITY_FACTORS = {
        'simple': -0.05,
        'medium': 0.0,
        'complex': 0.08,
        'very_complex': 0.12
    }
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def validate(self, query: str, response: str, worker_type: str = 'default', 
                 context: str = '') -> Dict:
        base_threshold = self.BASE_THRESHOLDS.get(worker_type.lower(), self.BASE_THRESHOLDS['default'])
        
        complexity = self._assess_complexity(query)
        complexity_adjustment = self.COMPLEXITY_FACTORS.get(complexity, 0.0)
        
        dynamic_threshold = base_threshold + complexity_adjustment
        dynamic_threshold = max(0.25, min(0.65, dynamic_threshold))
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        response_embedding = self.model.encode(response, convert_to_tensor=True)
        
        semantic_similarity = float(util.pytorch_cos_sim(query_embedding, response_embedding)[0][0])
        
        keyword_overlap = self._calculate_keyword_overlap(query, response)
        
        context_relevance = 0.5
        if context:
            context_embedding = self.model.encode(context, convert_to_tensor=True)
            context_relevance = float(util.pytorch_cos_sim(response_embedding, context_embedding)[0][0])
        
        combined_score = (semantic_similarity * 0.5 + keyword_overlap * 0.3 + context_relevance * 0.2)
        
        is_relevant = combined_score >= dynamic_threshold
        
        return {
            'is_relevant': is_relevant,
            'relevance_score': combined_score,
            'threshold_used': dynamic_threshold,
            'complexity': complexity,
            'breakdown': {
                'semantic_similarity': semantic_similarity,
                'keyword_overlap': keyword_overlap,
                'context_relevance': context_relevance
            }
        }
    
    def _assess_complexity(self, query: str) -> str:
        words = query.split()
        word_count = len(words)
        
        technical_terms = len(re.findall(r'\b(?:algorithm|function|optimize|analyze|implement|calculate|prove|verify|design|evaluate)\b', query.lower()))
        
        nested_structures = query.count('(') + query.count('[')
        
        conjunction_count = len(re.findall(r'\b(?:and|or|but|while|when|if|unless|although)\b', query.lower()))
        
        complexity_score = (
            (word_count / 50.0) * 0.3 +
            (technical_terms / 5.0) * 0.35 +
            (nested_structures / 3.0) * 0.2 +
            (conjunction_count / 3.0) * 0.15
        )
        
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.6:
            return 'medium'
        elif complexity_score < 0.85:
            return 'complex'
        else:
            return 'very_complex'
    
    def _calculate_keyword_overlap(self, query: str, response: str) -> float:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 
                     'being', 'this', 'that', 'these', 'those', 'it', 'its'}
        
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - stop_words
        response_words = set(re.findall(r'\b\w{3,}\b', response.lower())) - stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & response_words)
        overlap_ratio = overlap / len(query_words)
        
        return min(overlap_ratio, 1.0)
