import re
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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
    
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            self.cross_encoder = None
        
        self.tfidf = TfidfVectorizer(max_features=100, stop_words=None)
        self._performance_cache = {}
    
    def validate(self, query: str, response: str, worker_type: str = 'default', 
                 context: str = '') -> Dict:
        base_threshold = self.BASE_THRESHOLDS.get(worker_type.lower(), self.BASE_THRESHOLDS['default'])
        
        complexity = self._assess_complexity(query)
        complexity_adjustment = self._get_complexity_adjustment(complexity)
        
        dynamic_threshold = base_threshold + complexity_adjustment
        dynamic_threshold = max(0.25, min(0.70, dynamic_threshold))
        
        if self.cross_encoder:
            cross_score = self._cross_encoder_score(query, response)
            semantic_similarity = self._bi_encoder_similarity(query, response)
            combined_semantic = 0.7 * cross_score + 0.3 * semantic_similarity
        else:
            combined_semantic = self._bi_encoder_similarity(query, response)
        
        token_overlap = self._advanced_token_overlap(query, response)
        
        context_relevance = 0.5
        if context:
            context_relevance = self._contextual_relevance(query, response, context)
        
        combined_score = (
            combined_semantic * 0.50 + 
            token_overlap * 0.30 + 
            context_relevance * 0.20
        )
        
        is_relevant = combined_score >= dynamic_threshold
        
        return {
            'is_relevant': is_relevant,
            'relevance_score': combined_score,
            'threshold_used': dynamic_threshold,
            'complexity': complexity,
            'breakdown': {
                'semantic_score': combined_semantic,
                'token_overlap': token_overlap,
                'context_relevance': context_relevance
            }
        }
    
    def _cross_encoder_score(self, query: str, response: str) -> float:
        try:
            score = self.cross_encoder.predict([(query, response)])[0]
            return float(np.clip(score, 0.0, 1.0))
        except:
            return 0.5
    
    def _bi_encoder_similarity(self, query: str, response: str) -> float:
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        response_embedding = self.model.encode(response, convert_to_tensor=False)
        
        similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding) + 1e-9
        )
        return float(similarity)
    
    def _contextual_relevance(self, query: str, response: str, context: str) -> float:
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        response_embedding = self.model.encode(response, convert_to_tensor=False)
        context_embedding = self.model.encode(context, convert_to_tensor=False)
        
        qr_sim = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding) + 1e-9
        )
        
        rc_sim = np.dot(response_embedding, context_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(context_embedding) + 1e-9
        )
        
        return float(0.6 * qr_sim + 0.4 * rc_sim)
    
    def _assess_complexity(self, query: str) -> str:
        words = query.split()
        word_count = len(words)
        
        technical_patterns = [
            r'\b(algorithm|function|optimize|analyze|implement|calculate|prove|verify|design|evaluate)\b',
            r'\b(integrate|differentiate|solve|compute|determine|derive)\b',
            r'\b(class|method|object|interface|inheritance|polymorphism)\b'
        ]
        
        technical_terms = sum(len(re.findall(pattern, query.lower())) for pattern in technical_patterns)
        
        nested_structures = query.count('(') + query.count('[') + query.count('{')
        
        conjunctions = len(re.findall(
            r'\b(and|or|but|while|when|if|unless|although|because|since|whereas)\b',
            query.lower()
        ))
        
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        
        complexity_score = (
            (word_count / 50.0) * 0.25 +
            (technical_terms / 5.0) * 0.35 +
            (nested_structures / 3.0) * 0.20 +
            (conjunctions / 3.0) * 0.10 +
            (avg_word_length / 10.0) * 0.10
        )
        
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.6:
            return 'medium'
        elif complexity_score < 0.85:
            return 'complex'
        else:
            return 'very_complex'
    
    def _get_complexity_adjustment(self, complexity: str) -> float:
        adjustments = {
            'simple': -0.05,
            'medium': 0.0,
            'complex': 0.08,
            'very_complex': 0.12
        }
        return adjustments.get(complexity, 0.0)
    
    def _advanced_token_overlap(self, query: str, response: str) -> float:
        try:
            corpus = [query, response]
            self.tfidf.fit(corpus)
            query_vec = self.tfidf.transform([query]).toarray()[0]
            response_vec = self.tfidf.transform([response]).toarray()[0]
            
            overlap = np.dot(query_vec, response_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(response_vec) + 1e-9
            )
            return float(overlap)
        except:
            pass
        
        query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_tokens = set(re.findall(r'\b\w{3,}\b', response.lower()))
        
        if not query_tokens:
            return 0.0
        
        exact_overlap = len(query_tokens & response_tokens) / len(query_tokens)
        
        query_bigrams = set()
        response_bigrams = set()
        
        query_words = list(query_tokens)
        response_words = list(response_tokens)
        
        for i in range(len(query_words) - 1):
            query_bigrams.add(f"{query_words[i]}_{query_words[i+1]}")
        
        for i in range(len(response_words) - 1):
            response_bigrams.add(f"{response_words[i]}_{response_words[i+1]}")
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        return 0.7 * exact_overlap + 0.3 * bigram_overlap
