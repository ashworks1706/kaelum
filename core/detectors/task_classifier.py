from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.isotonic import IsotonicRegression
import time
from ..verification.threshold_calibrator import ThresholdCalibrator


class TaskClassifier:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self._calibrators = {}
        self._performance_history = {}
        self.threshold_calibrator = ThresholdCalibrator()
        
        self.task_profiles = {
            "code": {
                "debugging": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "Fix the bug in this code. Debug the error. Why is this not working?",
                        "There's an issue with my program. The code crashes when I run it.",
                        "Help me find the problem. Something is broken in this function.",
                        "This gives me an error. Can you identify what's wrong?",
                        "Stack trace shows error. Help debug this issue.",
                        "Why does this function throw an exception?"
                    ]
                },
                "optimization": {
                    "base_threshold": 0.55,
                    "exemplars": [
                        "Optimize this algorithm for better performance. Make this code faster.",
                        "Improve the efficiency of this function. Speed up the execution time.",
                        "This runs too slow. How can I make it more performant?",
                        "Reduce the time complexity. Make this more efficient.",
                        "Improve memory usage. Reduce computational cost.",
                        "Make this code run faster with better algorithms."
                    ]
                },
                "review": {
                    "base_threshold": 0.55,
                    "exemplars": [
                        "Review this code. Explain what this function does. Analyze this implementation.",
                        "What does this code do? Help me understand this logic.",
                        "Examine this function. Tell me how this works.",
                        "Code review needed. Explain the purpose of this method.",
                        "Analyze this implementation. What are the potential issues?",
                        "Help me understand what this code accomplishes."
                    ]
                },
                "testing": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "Write unit tests. Create test cases with assertions. Add pytest tests.",
                        "I need tests for this function. Generate test coverage.",
                        "Write test cases to verify this code. Add unit testing.",
                        "Create comprehensive tests. Add test assertions.",
                        "Generate test suite for this module with edge cases.",
                        "Add integration tests and mock dependencies."
                    ]
                },
                "algorithm": {
                    "base_threshold": 0.55,
                    "exemplars": [
                        "Implement a binary search algorithm. Create a graph data structure.",
                        "Write a sorting algorithm. Build a tree traversal method.",
                        "Implement dynamic programming solution. Create a hash table.",
                        "Design an algorithm to solve this problem efficiently.",
                        "Develop a recursive solution with memoization.",
                        "Create an algorithm using divide and conquer approach."
                    ]
                },
                "generation": {
                    "base_threshold": 0.45,
                    "exemplars": [
                        "Write a function to calculate. Create a class that handles. Generate code for.",
                        "Build a program that does. Develop a function for.",
                        "Create code to implement. Write a script that performs.",
                        "Generate a solution. Implement functionality to handle.",
                        "Develop a module that processes. Build a system for.",
                        "Code a solution that manages. Implement features for."
                    ]
                }
            },
            "factual": {
                "definition": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "What is photosynthesis? Define machine learning. What does entropy mean?",
                        "Explain the meaning of. What's the definition of. Define the term.",
                        "What does it mean when. Tell me what this concept is.",
                        "Give me the definition. Explain what this refers to.",
                        "Clarify the concept of. What exactly is meant by.",
                        "Provide the technical definition. Describe the meaning."
                    ]
                },
                "historical": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "When did World War II end? What year was the Renaissance? When did this happen?",
                        "What year did this occur? When was this invented? Tell me the date.",
                        "What time period was this? When did this event take place?",
                        "Give me the historical timeline. When in history did this happen?",
                        "What century did this occur? During which era?",
                        "Historical context of this event. Timeline of developments."
                    ]
                },
                "geographical": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "Where is Mount Everest? What is the capital of Japan? Where is this located?",
                        "Tell me the location. Where can I find this place? What's the geography?",
                        "Where is this situated? In which country is this? What's the address?",
                        "Point me to the location. Where exactly is this?",
                        "Geographic coordinates of. Location details.",
                        "Where on the map? In which region is this?"
                    ]
                },
                "quantitative": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "How many planets are there? What is the population of China? How much does it cost?",
                        "What's the total number? How much does this weigh? What's the quantity?",
                        "Give me the statistics. What are the numbers? How many are there?",
                        "What's the measurement? Provide the data. What's the amount?",
                        "Numerical value of. Calculate the total.",
                        "Statistical information about. Quantitative data on."
                    ]
                },
                "biographical": {
                    "base_threshold": 0.60,
                    "exemplars": [
                        "Who invented the telephone? Tell me about Marie Curie. Who discovered this?",
                        "Who is this person? What did they accomplish? Tell me about their life.",
                        "Who was responsible for this? Give me biographical information.",
                        "Tell me about this individual. Who created this?",
                        "Biography of this person. Life story and achievements.",
                        "Who made this discovery? Personal history of."
                    ]
                },
                "general": {
                    "base_threshold": 0.45,
                    "exemplars": [
                        "Explain the water cycle. Describe how airplanes fly. How does this work?",
                        "Tell me about this topic. Provide information on. Explain this concept.",
                        "Help me understand. Give me details about. What can you tell me?",
                        "Inform me about. Describe this process. Provide an explanation.",
                        "Give me an overview. Background information on.",
                        "Elaborate on this subject. Provide context for."
                    ]
                }
            }
        }
        
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self.exemplar_embeddings = {}
        
        for domain, tasks in self.task_profiles.items():
            self.exemplar_embeddings[domain] = {}
            for task_type, config in tasks.items():
                embeddings = self.encoder.encode(config['exemplars'], convert_to_tensor=False)
                self.exemplar_embeddings[domain][task_type] = embeddings
    
    def classify(self, query: str, domain: str) -> List[Tuple[str, float]]:
        if domain not in self.task_profiles:
            return [('general', 0.5)]
        
        query_embedding = self.encoder.encode(query, convert_to_tensor=False)
        
        results = {}
        
        for task_type, config in self.task_profiles[domain].items():
            exemplar_embeddings = self.exemplar_embeddings[domain][task_type]
            
            similarities = []
            for exemplar_emb in exemplar_embeddings:
                sim = np.dot(query_embedding, exemplar_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(exemplar_emb) + 1e-9
                )
                similarities.append(sim)
            
            max_sim = np.max(similarities)
            top_3_sims = sorted(similarities, reverse=True)[:3]
            avg_top3 = np.mean(top_3_sims)
            
            score = 0.6 * max_sim + 0.4 * avg_top3
            
            threshold = self._get_adaptive_threshold(domain, task_type, query)
            
            if score >= threshold:
                results[task_type] = float(score)
        
        if not results:
            default_task = 'generation' if domain == 'code' else 'general'
            return [(default_task, 0.5)]
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def record_outcome(self, domain: str, task_type: str, query: str, score: float, 
                      threshold: float, was_correct: bool):
        task_key = f"{domain}:{task_type}"
        self.threshold_calibrator.record_decision(
            score=score,
            threshold=threshold,
            actual_result=was_correct,
            task_type=task_key,
            timestamp=time.time()
        )
        self.update_performance(domain, task_type, query, was_correct)
    
    def _get_adaptive_threshold(self, domain: str, task_type: str, query: str) -> float:
        task_key = f"{domain}:{task_type}"
        optimal_threshold = self.threshold_calibrator.get_optimal_threshold(task_key)
        
        if optimal_threshold != 0.5:
            return optimal_threshold
        
        base_threshold = self.task_profiles[domain][task_type]['base_threshold']
        
        words = query.split()
        word_count = len(words)
        
        if word_count < 5:
            adjustment = 0.10
        elif word_count < 10:
            adjustment = 0.05
        elif word_count > 30:
            adjustment = -0.05
        else:
            adjustment = 0.0
        
        key = f"{domain}:{task_type}"
        if key in self._performance_history:
            history = self._performance_history[key]
            if len(history) >= 5:
                avg_performance = np.mean([h['success'] for h in history[-10:]])
                if avg_performance < 0.5:
                    adjustment += 0.05
                elif avg_performance > 0.8:
                    adjustment -= 0.03
        
        return max(0.30, min(0.80, base_threshold + adjustment))
    
    def update_performance(self, domain: str, task_type: str, query: str, success: bool):
        key = f"{domain}:{task_type}"
        if key not in self._performance_history:
            self._performance_history[key] = []
        
        self._performance_history[key].append({
            'query': query,
            'success': success,
            'timestamp': np.datetime64('now')
        })
        
        if len(self._performance_history[key]) > 100:
            self._performance_history[key] = self._performance_history[key][-100:]
    
    def classify_single(self, query: str, domain: str) -> Dict[str, any]:
        results = self.classify(query, domain)
        
        if not results:
            return {'task': 'general', 'confidence': 0.5, 'alternatives': []}
        
        primary_task, primary_score = results[0]
        
        is_ambiguous = len(results) > 1 and results[1][1] > primary_score - 0.12
        
        alternatives = results[1:3] if len(results) > 1 else []
        
        return {
            'task': primary_task,
            'confidence': primary_score,
            'ambiguous': is_ambiguous,
            'alternatives': alternatives
        }
