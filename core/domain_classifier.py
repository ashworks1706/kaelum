from typing import Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class DomainClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.domain_prototypes = {
            'code': [
                'implement a function that sorts an array',
                'write code to connect to database',
                'create a class for user authentication',
                'debug this program error',
                'optimize the algorithm performance',
                'refactor the code structure',
                'add error handling to the function',
                'write unit tests for the module'
            ],
            'math': [
                'calculate the derivative of function',
                'solve this equation for x',
                'find the integral of expression',
                'compute the probability distribution',
                'determine the matrix eigenvalues',
                'calculate the sum of series',
                'find the limit as x approaches infinity',
                'solve the differential equation'
            ],
            'logic': [
                'prove this logical statement',
                'determine if the argument is valid',
                'check if the conclusion follows from premises',
                'verify the logical equivalence',
                'construct a truth table',
                'identify the logical fallacy',
                'derive the conclusion using modus ponens',
                'show that the implication holds'
            ],
            'factual': [
                'what is the capital of country',
                'when did the event occur',
                'who invented this technology',
                'where is the location situated',
                'explain what this term means',
                'describe the historical significance',
                'list the characteristics of species',
                'define the scientific concept'
            ],
            'creative': [
                'write a story about adventure',
                'create a poem about nature',
                'brainstorm ideas for campaign',
                'design a logo concept',
                'compose a song lyrics',
                'imagine a fictional world',
                'develop a character backstory',
                'craft a compelling narrative'
            ],
            'analysis': [
                'analyze the trends in data',
                'compare these two approaches',
                'evaluate the effectiveness of method',
                'assess the impact of decision',
                'interpret the results of study',
                'examine the relationship between variables',
                'critique the argument presented',
                'review the strengths and weaknesses'
            ]
        }
        
        self._prototype_embeddings = {}
        for domain, examples in self.domain_prototypes.items():
            embeddings = self.encoder.encode(examples, convert_to_tensor=False)
            self._prototype_embeddings[domain] = embeddings
    
    def classify_domain(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)[0]
        
        domain_scores = {}
        for domain, prototype_embeddings in self._prototype_embeddings.items():
            similarities = []
            for proto_emb in prototype_embeddings:
                sim = np.dot(query_embedding, proto_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(proto_emb)
                )
                similarities.append(sim)
            
            max_sim = float(np.max(similarities))
            avg_sim = float(np.mean(similarities))
            domain_scores[domain] = 0.7 * max_sim + 0.3 * avg_sim
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain[0], best_domain[1], domain_scores
    
    def get_domain_features(self, query: str) -> Dict[str, float]:
        _, _, domain_scores = self.classify_domain(query)
        
        features = {}
        for domain, score in domain_scores.items():
            features[f'has_{domain}_domain'] = score
        
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        features['domain_confidence'] = sorted_domains[0][1]
        features['domain_ambiguity'] = sorted_domains[0][1] - sorted_domains[1][1] if len(sorted_domains) > 1 else 1.0
        
        return features
