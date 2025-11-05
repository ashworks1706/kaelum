from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util


class TaskClassifier:
    CONFIGS = {
        "code": {
            "debugging": {
                "threshold": 0.65,
                "keywords": ['debug', 'fix', 'error', 'bug', 'issue', 'problem', 'broken'],
                "exemplar": "Fix the bug in this code. Debug the error. Why is this not working?"
            },
            "optimization": {
                "threshold": 0.60,
                "keywords": ['optimize', 'improve', 'faster', 'performance', 'efficient', 'speed'],
                "exemplar": "Optimize this algorithm for better performance. Make this code faster and more efficient."
            },
            "review": {
                "threshold": 0.60,
                "keywords": ['review', 'analyze', 'explain', 'understand', 'what does', 'how does'],
                "exemplar": "Review this code. Explain what this function does. Analyze this implementation."
            },
            "testing": {
                "threshold": 0.65,
                "keywords": ['test', 'unittest', 'pytest', 'jest', 'assert', 'coverage'],
                "exemplar": "Write unit tests. Create test cases with assertions. Add pytest tests."
            },
            "algorithm": {
                "threshold": 0.60,
                "keywords": ['algorithm', 'data structure', 'implement', 'sort', 'search', 'tree', 'graph'],
                "exemplar": "Implement a binary search algorithm. Create a graph data structure."
            },
            "generation": {
                "threshold": 0.50,
                "keywords": ['write', 'create', 'generate', 'build', 'make'],
                "exemplar": "Write a function to calculate. Create a class that handles. Generate code for."
            }
        },
        "factual": {
            "definition": {
                "threshold": 0.65,
                "keywords": ['what is', 'define', 'definition', 'meaning', 'means'],
                "exemplar": "What is photosynthesis? Define machine learning. What does entropy mean?"
            },
            "historical": {
                "threshold": 0.65,
                "keywords": ['when', 'year', 'date', 'history', 'began', 'ended', 'happened'],
                "exemplar": "When did World War II end? What year was the Renaissance? When did this happen?"
            },
            "geographical": {
                "threshold": 0.65,
                "keywords": ['where', 'location', 'located', 'place', 'city', 'country', 'capital'],
                "exemplar": "Where is Mount Everest? What is the capital of Japan? Where is this located?"
            },
            "quantitative": {
                "threshold": 0.65,
                "keywords": ['how many', 'how much', 'number', 'amount', 'quantity', 'population'],
                "exemplar": "How many planets are there? What is the population of China? How much does it cost?"
            },
            "biographical": {
                "threshold": 0.65,
                "keywords": ['who', 'invented', 'discovered', 'created', 'person', 'biography'],
                "exemplar": "Who invented the telephone? Tell me about Marie Curie. Who discovered this?"
            },
            "general": {
                "threshold": 0.50,
                "keywords": ['explain', 'describe', 'how', 'why', 'tell me about'],
                "exemplar": "Explain the water cycle. Describe how airplanes fly. How does this work?"
            }
        }
    }
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.exemplar_embeddings = {}
        
        for domain, tasks in self.CONFIGS.items():
            self.exemplar_embeddings[domain] = {}
            for task_type, config in tasks.items():
                embedding = self.encoder.encode(config['exemplar'], convert_to_tensor=True)
                self.exemplar_embeddings[domain][task_type] = embedding
    
    def classify(self, query: str, domain: str) -> List[Tuple[str, float]]:
        if domain not in self.CONFIGS:
            return [('general', 0.5)]
        
        query_lower = query.lower()
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        results = {}
        
        for task_type, config in self.CONFIGS[domain].items():
            score = 0.0
            
            # 1. Semantic similarity (weight: 0.6)
            exemplar_emb = self.exemplar_embeddings[domain][task_type]
            semantic_score = float(util.cos_sim(query_embedding, exemplar_emb)[0][0])
            score += semantic_score * 0.6
            
            # 2. Keyword matching (weight: 0.4)
            keyword_hits = sum(1 for kw in config['keywords'] if kw in query_lower)
            keyword_score = min(keyword_hits * 0.15, 0.4)
            score += keyword_score
            
            if score >= config['threshold']:
                results[task_type] = score
        
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
        
        is_ambiguous = len(results) > 1 and results[1][1] > primary_score - 0.15
        
        alternatives = results[1:3] if len(results) > 1 else []
        
        return {
            'task': primary_task,
            'confidence': primary_score,
            'ambiguous': is_ambiguous,
            'alternatives': alternatives
        }
