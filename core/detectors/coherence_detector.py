from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time
from ..verification.threshold_calibrator import ThresholdCalibrator


class CoherenceDetector:
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        try:
            self.nli_pipeline = pipeline('text-classification', 
                                         model='facebook/bart-large-mnli',
                                         device=-1)
        except:
            self.nli_pipeline = None
        self.threshold_calibrator = ThresholdCalibrator()
    
    def assess_coherence(self, text: str, task_type: str = 'general') -> Dict[str, float]:
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {
                'overall_coherence': 0.5,
                'sentence_continuity': 0.5,
                'topic_consistency': 0.5,
                'logical_flow': 0.5
            }
        
        continuity_score = self._measure_sentence_continuity(sentences)
        topic_score = self._measure_topic_consistency(sentences)
        flow_score = self._measure_logical_flow(sentences)
        
        task_weights = self._get_task_weights(task_type)
        overall = (continuity_score * task_weights['continuity'] +
                  topic_score * task_weights['topic'] +
                  flow_score * task_weights['flow'])
        
        return {
            'overall_coherence': overall,
            'sentence_continuity': continuity_score,
            'topic_consistency': topic_score,
            'logical_flow': flow_score
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences
    
    def _measure_sentence_continuity(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        
        if not self.nli_pipeline:
            embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
            continuity_scores = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-9
                )
                continuity_scores.append(sim)
            return float(np.mean(continuity_scores)) if continuity_scores else 0.5
        
        continuity_scores = []
        for i in range(len(sentences) - 1):
            premise = sentences[i]
            hypothesis = sentences[i + 1]
            
            try:
                result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")[0]
                if result['label'] == 'ENTAILMENT':
                    continuity_scores.append(result['score'])
                elif result['label'] == 'NEUTRAL':
                    continuity_scores.append(result['score'] * 0.5)
                else:
                    continuity_scores.append(0.0)
            except:
                continuity_scores.append(0.5)
        
        return float(np.mean(continuity_scores)) if continuity_scores else 0.5
    
    def _measure_topic_consistency(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        
        embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        if not similarities:
            return 0.5
        
        avg_similarity = float(np.mean(similarities))
        return avg_similarity
    
    def _measure_logical_flow(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        
        embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
        
        sequential_similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            sequential_similarities.append(sim)
        
        flow_score = float(np.mean(sequential_similarities))
        
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently',
                          'thus', 'hence', 'additionally', 'meanwhile', 'nevertheless',
                          'then', 'next', 'finally', 'first', 'second', 'lastly']
        
        text_lower = ' '.join(sentences).lower()
        transition_count = sum(1 for tw in transition_words if tw in text_lower)
        transition_bonus = min(transition_count * 0.05, 0.15)
        
        return min(flow_score + transition_bonus, 1.0)
    
    def _get_task_weights(self, task_type: str) -> Dict[str, float]:
        weights = {
            'storytelling': {'continuity': 0.4, 'topic': 0.3, 'flow': 0.3},
            'poetry': {'continuity': 0.3, 'topic': 0.4, 'flow': 0.3},
            'writing': {'continuity': 0.35, 'topic': 0.3, 'flow': 0.35},
            'ideation': {'continuity': 0.2, 'topic': 0.5, 'flow': 0.3},
            'design': {'continuity': 0.3, 'topic': 0.4, 'flow': 0.3},
            'dialogue': {'continuity': 0.4, 'topic': 0.25, 'flow': 0.35},
            'general': {'continuity': 0.35, 'topic': 0.35, 'flow': 0.3}
        }
        return weights.get(task_type, weights['general'])
    
    def record_outcome(self, task_type: str, score: float, threshold: float, was_correct: bool):
        self.threshold_calibrator.record_decision(
            score=score,
            threshold=threshold,
            actual_result=was_correct,
            task_type=f"coherence:{task_type}",
            timestamp=time.time()
        )
