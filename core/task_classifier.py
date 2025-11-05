from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np


class TaskClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.task_profiles = {
            "code": {
                "debugging": {
                    "threshold": 0.60,
                    "exemplars": [
                        "Fix the bug in this code. Debug the error. Why is this not working?",
                        "There's an issue with my program. The code crashes when I run it.",
                        "Help me find the problem. Something is broken in this function.",
                        "This gives me an error. Can you identify what's wrong?"
                    ]
                },
                "optimization": {
                    "threshold": 0.55,
                    "exemplars": [
                        "Optimize this algorithm for better performance. Make this code faster.",
                        "Improve the efficiency of this function. Speed up the execution time.",
                        "This runs too slow. How can I make it more performant?",
                        "Reduce the time complexity. Make this more efficient."
                    ]
                },
                "review": {
                    "threshold": 0.55,
                    "exemplars": [
                        "Review this code. Explain what this function does. Analyze this implementation.",
                        "What does this code do? Help me understand this logic.",
                        "Examine this function. Tell me how this works.",
                        "Code review needed. Explain the purpose of this method."
                    ]
                },
                "testing": {
                    "threshold": 0.60,
                    "exemplars": [
                        "Write unit tests. Create test cases with assertions. Add pytest tests.",
                        "I need tests for this function. Generate test coverage.",
                        "Write test cases to verify this code. Add unit testing.",
                        "Create comprehensive tests. Add test assertions."
                    ]
                },
                "algorithm": {
                    "threshold": 0.55,
                    "exemplars": [
                        "Implement a binary search algorithm. Create a graph data structure.",
                        "Write a sorting algorithm. Build a tree traversal method.",
                        "Implement dynamic programming solution. Create a hash table.",
                        "Design an algorithm to solve this problem efficiently."
                    ]
                },
                "generation": {
                    "threshold": 0.45,
                    "exemplars": [
                        "Write a function to calculate. Create a class that handles. Generate code for.",
                        "Build a program that does. Develop a function for.",
                        "Create code to implement. Write a script that performs.",
                        "Generate a solution. Implement functionality to handle."
                    ]
                }
            },
            "factual": {
                "definition": {
                    "threshold": 0.60,
                    "exemplars": [
                        "What is photosynthesis? Define machine learning. What does entropy mean?",
                        "Explain the meaning of. What's the definition of. Define the term.",
                        "What does it mean when. Tell me what this concept is.",
                        "Give me the definition. Explain what this refers to."
                    ]
                },
                "historical": {
                    "threshold": 0.60,
                    "exemplars": [
                        "When did World War II end? What year was the Renaissance? When did this happen?",
                        "What year did this occur? When was this invented? Tell me the date.",
                        "What time period was this? When did this event take place?",
                        "Give me the historical timeline. When in history did this happen?"
                    ]
                },
                "geographical": {
                    "threshold": 0.60,
                    "exemplars": [
                        "Where is Mount Everest? What is the capital of Japan? Where is this located?",
                        "Tell me the location. Where can I find this place? What's the geography?",
                        "Where is this situated? In which country is this? What's the address?",
                        "Point me to the location. Where exactly is this?"
                    ]
                },
                "quantitative": {
                    "threshold": 0.60,
                    "exemplars": [
                        "How many planets are there? What is the population of China? How much does it cost?",
                        "What's the total number? How much does this weigh? What's the quantity?",
                        "Give me the statistics. What are the numbers? How many are there?",
                        "What's the measurement? Provide the data. What's the amount?"
                    ]
                },
                "biographical": {
                    "threshold": 0.60,
                    "exemplars": [
                        "Who invented the telephone? Tell me about Marie Curie. Who discovered this?",
                        "Who is this person? What did they accomplish? Tell me about their life.",
                        "Who was responsible for this? Give me biographical information.",
                        "Tell me about this individual. Who created this?"
                    ]
                },
                "general": {
                    "threshold": 0.45,
                    "exemplars": [
                        "Explain the water cycle. Describe how airplanes fly. How does this work?",
                        "Tell me about this topic. Provide information on. Explain this concept.",
                        "Help me understand. Give me details about. What can you tell me?",
                        "Inform me about. Describe this process. Provide an explanation."
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
            avg_top2 = np.mean(sorted(similarities, reverse=True)[:2])
            
            score = 0.7 * max_sim + 0.3 * avg_top2
            
            if score >= config['threshold']:
                results[task_type] = float(score)
        
        if not results:
            default_task = 'generation' if domain == 'code' else 'general'
            return [(default_task, 0.5)]
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
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
