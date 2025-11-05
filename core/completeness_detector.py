from typing import Dict, List
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np


class CompletenessDetector:
    
    def __init__(self):
        self.nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    
    def is_complete(self, query: str, response: str, context: List[str] = None) -> Dict:
        context = context or []
        
        nli_score = self._check_entailment(query, response)
        semantic_coverage = self._semantic_coverage(query, response)
        aspect_coverage = self._aspect_based_coverage(query, response)
        has_dangling = self._check_dangling_references(response, context)
        coherence_score = self._check_coherence(response)
        
        completeness_score = (
            nli_score * 0.30 +
            semantic_coverage * 0.30 +
            aspect_coverage * 0.25 +
            coherence_score * 0.15
        )
        
        if has_dangling:
            completeness_score *= 0.8
        
        is_complete = completeness_score >= 0.65
        
        missing_aspects = self._identify_missing_aspects(query, response)
        
        return {
            'is_complete': is_complete,
            'confidence': completeness_score,
            'nli_score': nli_score,
            'semantic_coverage': semantic_coverage,
            'aspect_coverage': aspect_coverage,
            'has_dangling_references': has_dangling,
            'coherence_score': coherence_score,
            'missing_aspects': missing_aspects
        }
    
    def _check_entailment(self, query: str, response: str) -> float:
        try:
            hypothesis = f"This text fully answers: {query}"
            result = self.nli_pipeline(f"{response}", hypothesis)
            
            if result[0]['label'] == 'ENTAILMENT':
                return result[0]['score']
            elif result[0]['label'] == 'NEUTRAL':
                return result[0]['score'] * 0.5
            else:
                return 0.0
        except:
            return 0.0
    
    def _semantic_coverage(self, query: str, response: str) -> float:
        query_embedding = self.encoder.encode(query, convert_to_tensor=False)
        response_embedding = self.encoder.encode(response, convert_to_tensor=False)
        
        similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding) + 1e-9
        )
        
        query_sentences = [s.strip() for s in query.split('.') if s.strip()]
        if len(query_sentences) <= 1:
            return float(similarity)
        
        response_emb = response_embedding
        sentence_coverages = []
        
        for sentence in query_sentences:
            sent_emb = self.encoder.encode(sentence, convert_to_tensor=False)
            sim = np.dot(sent_emb, response_emb) / (
                np.linalg.norm(sent_emb) * np.linalg.norm(response_emb) + 1e-9
            )
            sentence_coverages.append(sim)
        
        avg_coverage = np.mean(sentence_coverages)
        return float(0.6 * similarity + 0.4 * avg_coverage)
    
    def _aspect_based_coverage(self, query: str, response: str) -> float:
        aspects = self._extract_aspects(query)
        
        if not aspects:
            return 0.8
        
        try:
            covered = 0
            for aspect in aspects:
                result = self.qa_model(question=aspect, context=response)
                if result['score'] > 0.3:
                    covered += 1
            
            return covered / len(aspects)
        except:
            return 0.5
    
    def _extract_aspects(self, query: str) -> List[str]:
        aspects = []
        
        query_lower = query.lower()
        
        question_patterns = {
            'what': "What is",
            'why': "Why does",
            'how': "How does",
            'when': "When did",
            'where': "Where is",
            'who': "Who did"
        }
        
        for qw, prefix in question_patterns.items():
            if qw in query_lower:
                aspects.append(f"{prefix} this happen")
        
        if 'and' in query or '&' in query:
            parts = query.replace('&', 'and').split('and')
            aspects.extend([p.strip() for p in parts if len(p.strip()) > 5])
        
        if not aspects:
            aspects = [query]
        
        return aspects[:5]
    
    def _check_coherence(self, response: str) -> float:
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
        
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-9
            )
            coherence_scores.append(sim)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.5
    
    def _check_dangling_references(self, response: str, context: List[str]) -> bool:
        if not context:
            reference_patterns = [
                r'\bas\s+(mentioned|stated|shown|discussed)\b',
                r'\bsee\s+(above|below|earlier|previous)\b',
                r'\brefer\s+to\b',
                r'\b(this|that|these|those)\s+(?:means|refers|indicates)\b'
            ]
            
            import re
            response_lower = response.lower()
            for pattern in reference_patterns:
                if re.search(pattern, response_lower):
                    return True
        
        return False
    
    def _identify_missing_aspects(self, query: str, response: str) -> List[str]:
        missing = []
        
        aspects = self._extract_aspects(query)
        
        try:
            for aspect in aspects:
                result = self.qa_model(question=aspect, context=response)
                if result['score'] < 0.3:
                    missing.append(aspect)
        except:
            pass
        
        return missing[:3]
