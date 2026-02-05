from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from ..verification.threshold_calibrator import ThresholdCalibrator

class WorkerTypeClassifier:
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.threshold_calibrator = ThresholdCalibrator()
        
        self.worker_profiles = {
            'math': {
                'threshold': 0.50,
                'exemplars': [
                    'solve the equation 2x + 5 = 15 for x',
                    'find the derivative of x^2 + 3x with respect to x',
                    'calculate the integral of sin(x) from 0 to pi',
                    'prove that the sum of angles in a triangle is 180 degrees',
                    'compute the determinant of this matrix',
                    'find the roots of the quadratic equation'
                ]
            },
            'code': {
                'threshold': 0.50,
                'exemplars': [
                    'write a function to sort an array using quicksort',
                    'implement binary search in python for a sorted list',
                    'debug this code that crashes when processing input',
                    'optimize this algorithm for better time complexity',
                    'refactor this class to follow SOLID principles',
                    'create a REST API endpoint with authentication'
                ]
            },
            'logic': {
                'threshold': 0.50,
                'exemplars': [
                    'if all humans are mortal and socrates is human then what follows',
                    'determine if this argument is logically valid',
                    'what conclusion follows from these premises using modus ponens',
                    'identify the logical fallacy in this reasoning',
                    'prove this statement using formal logic',
                    'construct a truth table for this logical expression'
                ]
            },
            'creative': {
                'threshold': 0.45,
                'exemplars': [
                    'write a poem about nature and the changing seasons',
                    'create a short story about adventure in space',
                    'brainstorm innovative ideas for a new product launch',
                    'compose a haiku capturing the essence of autumn',
                    'design a unique logo concept for a tech startup',
                    'generate creative names for a coffee shop'
                ]
            },
            'factual': {
                'threshold': 0.50,
                'exemplars': [
                    'what is the capital of france and its population',
                    'who invented the telephone and in what year',
                    'explain how photosynthesis works in plants',
                    'when did world war 2 end and what were the outcomes',
                    'where is mount everest located and how tall is it',
                    'define quantum mechanics and its key principles'
                ]
            },
            'analysis': {
                'threshold': 0.48,
                'exemplars': [
                    'analyze the themes in shakespeares hamlet',
                    'compare renewable energy and fossil fuels comprehensively',
                    'evaluate the effectiveness of this marketing strategy',
                    'examine the causes of the 2008 financial crisis',
                    'assess the impact of social media on society',
                    'critique the argument presented in this article'
                ]
            }
        }
        
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self.worker_embeddings = {}
        for worker, profile in self.worker_profiles.items():
            embeddings = self.model.encode(profile['exemplars'], convert_to_tensor=False)
            self.worker_embeddings[worker] = embeddings
    
    def classify_worker(self, query: str) -> Dict:
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        scores = {}
        
        for worker, profile in self.worker_profiles.items():
            exemplar_embeddings = self.worker_embeddings[worker]
            
            similarities = []
            for exemplar_emb in exemplar_embeddings:
                sim = np.dot(query_embedding, exemplar_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(exemplar_emb) + 1e-9
                )
                similarities.append(sim)
            
            max_sim = np.max(similarities)
            avg_top3 = np.mean(sorted(similarities, reverse=True)[:3])
            
            combined_score = 0.7 * max_sim + 0.3 * avg_top3
            scores[worker] = float(combined_score)
        
        sorted_workers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_worker, top_score = sorted_workers[0]
        second_score = sorted_workers[1][1] if len(sorted_workers) > 1 else 0.0
        
        is_ambiguous = (top_score - second_score) < 0.12
        
        alternatives = [w for w, s in sorted_workers[1:4] if s > 0.35]
        
        return {
            'worker': top_worker,
            'confidence': top_score,
            'is_ambiguous': is_ambiguous,
            'alternatives': alternatives,
            'all_scores': dict(sorted_workers)
        }
    
    def record_outcome(self, worker: str, score: float, threshold: float, was_correct: bool):
        self.threshold_calibrator.record_decision(
            score=score,
            threshold=threshold,
            actual_result=was_correct,
            task_type=f"worker:{worker}",
            timestamp=time.time()
        )
    
    def get_optimal_threshold(self, worker: str) -> float:
        optimal = self.threshold_calibrator.get_optimal_threshold(
            f"worker:{worker}",
            default=self.worker_profiles.get(worker, {}).get('threshold', 0.5)
        )
        return optimal
