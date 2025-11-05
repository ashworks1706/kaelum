from typing import Dict, List
from sentence_transformers import SentenceTransformer, util


class WorkerTypeClassifier:
    
    WORKER_PROFILES = {
        'math': {
            'keywords': ['calculate', 'solve', 'equation', 'derivative', 'integral', 'algebra', 'trigonometry', 'calculus', 'proof', 'theorem'],
            'patterns': ['x =', 'f(x)', 'dx', 'dy', '∫', '∑', '∏', '√'],
            'examples': [
                'solve the equation 2x + 5 = 15',
                'find the derivative of x^2 + 3x',
                'calculate the integral of sin(x)',
                'prove that the sum of angles in a triangle is 180'
            ]
        },
        'code': {
            'keywords': ['function', 'class', 'algorithm', 'implement', 'debug', 'optimize', 'refactor', 'test', 'code', 'program'],
            'patterns': ['def ', 'class ', 'function ', 'import ', 'return ', '{', '}', '()', '[]'],
            'examples': [
                'write a function to sort an array',
                'implement binary search in python',
                'debug this code that crashes',
                'optimize this algorithm for better performance'
            ]
        },
        'logic': {
            'keywords': ['if', 'then', 'therefore', 'premise', 'conclusion', 'implies', 'entails', 'valid', 'sound', 'fallacy'],
            'patterns': ['→', '∧', '∨', '¬', '∀', '∃', 'if...then', 'all...are'],
            'examples': [
                'if all humans are mortal and socrates is human then what',
                'determine if this argument is valid',
                'what conclusion follows from these premises',
                'identify the logical fallacy in this reasoning'
            ]
        },
        'creative': {
            'keywords': ['write', 'create', 'story', 'poem', 'imagine', 'brainstorm', 'generate', 'compose', 'craft', 'design'],
            'patterns': ['write a', 'create a', 'compose a', 'imagine a', 'tell me a'],
            'examples': [
                'write a poem about nature',
                'create a short story about adventure',
                'brainstorm ideas for a new product',
                'compose a haiku about seasons'
            ]
        },
        'factual': {
            'keywords': ['what', 'who', 'when', 'where', 'define', 'explain', 'describe', 'tell', 'information', 'fact'],
            'patterns': ['what is', 'who was', 'when did', 'where is', 'define ', 'explain '],
            'examples': [
                'what is the capital of france',
                'who invented the telephone',
                'explain how photosynthesis works',
                'when did world war 2 end'
            ]
        },
        'analysis': {
            'keywords': ['analyze', 'compare', 'evaluate', 'assess', 'examine', 'investigate', 'review', 'critique', 'interpret'],
            'patterns': ['compare ', 'analyze ', 'evaluate ', 'assess '],
            'examples': [
                'analyze the themes in this text',
                'compare renewable and fossil fuels',
                'evaluate the effectiveness of this approach',
                'examine the causes of inflation'
            ]
        }
    }
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self.worker_embeddings = {}
        for worker, profile in self.WORKER_PROFILES.items():
            combined_text = ' '.join(profile['examples'] + profile['keywords'])
            self.worker_embeddings[worker] = self.model.encode(combined_text, convert_to_tensor=True)
    
    def classify_worker(self, query: str) -> Dict:
        query_lower = query.lower()
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        scores = {}
        
        for worker, profile in self.WORKER_PROFILES.items():
            semantic_sim = float(util.pytorch_cos_sim(query_embedding, self.worker_embeddings[worker])[0][0])
            
            keyword_matches = sum(1 for kw in profile['keywords'] if kw in query_lower)
            keyword_score = min(keyword_matches / 5.0, 1.0)
            
            pattern_matches = sum(1 for pattern in profile['patterns'] if pattern.lower() in query_lower)
            pattern_score = min(pattern_matches / 3.0, 1.0)
            
            combined_score = (semantic_sim * 0.6 + keyword_score * 0.25 + pattern_score * 0.15)
            scores[worker] = combined_score
        
        sorted_workers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_worker, top_score = sorted_workers[0]
        second_score = sorted_workers[1][1] if len(sorted_workers) > 1 else 0.0
        
        is_ambiguous = (top_score - second_score) < 0.15
        
        alternatives = [w for w, s in sorted_workers[1:4] if s > 0.3]
        
        return {
            'worker': top_worker,
            'confidence': top_score,
            'is_ambiguous': is_ambiguous,
            'alternatives': alternatives,
            'all_scores': dict(sorted_workers)
        }
