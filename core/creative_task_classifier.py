from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class CreativeTaskClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.task_profiles = {
            'storytelling': {
                'threshold': 0.55,
                'exemplars': [
                    'Write a story about a detective solving a mystery in Victorian London',
                    'Create a narrative about space exploration and first contact with aliens',
                    'Tell me a tale of adventure featuring a young hero on a quest',
                    'Develop a plot for a mystery novel set in a small coastal town',
                    'Write a short story with a twist ending about time travel'
                ]
            },
            'poetry': {
                'threshold': 0.60,
                'exemplars': [
                    'Write a poem about nature and the changing seasons',
                    'Create a haiku about the beauty of autumn leaves falling',
                    'Compose a sonnet about love and longing across distance',
                    'Make a limerick about a mischievous cat and its adventures',
                    'Write free verse poetry exploring themes of identity'
                ]
            },
            'writing': {
                'threshold': 0.50,
                'exemplars': [
                    'Write an essay on climate change and its global impact',
                    'Create a blog post about technology trends in healthcare',
                    'Draft an article about economic development in emerging markets',
                    'Compose a report on market trends and consumer behavior',
                    'Write a persuasive piece about the importance of education'
                ]
            },
            'ideation': {
                'threshold': 0.55,
                'exemplars': [
                    'Brainstorm ideas for a tech startup in the education space',
                    'Suggest concepts for a viral marketing campaign targeting millennials',
                    'Propose innovative solutions to reduce urban traffic congestion',
                    'Generate ideas for new product features in mobile apps',
                    'Think of creative ways to improve remote team collaboration'
                ]
            },
            'design': {
                'threshold': 0.55,
                'exemplars': [
                    'Design a user interface for a mobile banking application',
                    'Plan the architecture of a scalable microservices system',
                    'Create a layout for a modern minimalist website homepage',
                    'Develop a framework for organizing large software projects',
                    'Design a visual identity system for a new brand'
                ]
            },
            'dialogue': {
                'threshold': 0.60,
                'exemplars': [
                    'Write dialogue between two characters arguing about philosophy',
                    'Create a conversation script for a job interview scenario',
                    'Develop a chat exchange between friends planning a vacation',
                    'Script a discussion between experts debating AI ethics',
                    'Write natural dialogue for a romantic comedy scene'
                ]
            },
            'general_creative': {
                'threshold': 0.45,
                'exemplars': [
                    'Be creative with this task and think outside the box',
                    'Use your imagination to come up with something unique',
                    'Think of something original and innovative for this project',
                    'Create something unique that stands out from conventional approaches',
                    'Generate an inspired and artistic solution to this challenge'
                ]
            }
        }
        
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self._embedding_cache = {}
        for task, profile in self.task_profiles.items():
            embeddings = self.encoder.encode(profile['exemplars'], convert_to_tensor=False)
            self._embedding_cache[task] = embeddings
    
    def classify_task(self, query: str) -> Tuple[str, float, bool, Dict[str, float]]:
        query_embedding = self.encoder.encode(query, convert_to_tensor=False)
        
        all_scores = {}
        
        for task, profile in self.task_profiles.items():
            exemplar_embeddings = self._embedding_cache[task]
            
            similarities = []
            for exemplar_emb in exemplar_embeddings:
                sim = np.dot(query_embedding, exemplar_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(exemplar_emb) + 1e-9
                )
                similarities.append(sim)
            
            max_similarity = np.max(similarities)
            avg_top3 = np.mean(sorted(similarities, reverse=True)[:3])
            
            score = 0.65 * max_similarity + 0.35 * avg_top3
            
            all_scores[task] = float(score)
        
        sorted_tasks = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_task, best_score = sorted_tasks[0]
        second_score = sorted_tasks[1][1] if len(sorted_tasks) > 1 else 0.0
        
        is_ambiguous = (best_score - second_score) < 0.12
        
        alternatives = {task: score for task, score in sorted_tasks[1:4] if score > 0.35}
        
        return best_task, best_score, is_ambiguous, alternatives
