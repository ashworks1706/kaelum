from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class CreativeTaskClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.task_profiles = {
            'storytelling': {
                'keywords': ['story', 'narrative', 'fiction', 'tale', 'plot', 'character', 
                           'novel', 'chapter', 'protagonist', 'setting'],
                'patterns': [r'\b(write|tell|create).*story\b', r'\bnarrative\b', 
                           r'\bfiction\b', r'\btale\b', r'\bplot\b'],
                'examples': [
                    'Write a story about a detective',
                    'Create a narrative about space exploration',
                    'Tell me a tale of adventure',
                    'Develop a plot for a mystery novel'
                ]
            },
            'poetry': {
                'keywords': ['poem', 'verse', 'rhyme', 'haiku', 'sonnet', 'stanza',
                           'lyric', 'ballad', 'limerick', 'ode'],
                'patterns': [r'\bpoem\b', r'\bverse\b', r'\brhyme\b', r'\bhaiku\b',
                           r'\bsonnet\b', r'\bstanza\b'],
                'examples': [
                    'Write a poem about nature',
                    'Create a haiku about seasons',
                    'Compose a sonnet about love',
                    'Make a limerick about cats'
                ]
            },
            'writing': {
                'keywords': ['essay', 'article', 'blog', 'post', 'write', 'compose',
                           'draft', 'paper', 'report', 'document'],
                'patterns': [r'\bessay\b', r'\barticle\b', r'\bblog\b', r'\bpost\b',
                           r'\breport\b', r'\bdocument\b'],
                'examples': [
                    'Write an essay on climate change',
                    'Create a blog post about technology',
                    'Draft an article about economics',
                    'Compose a report on market trends'
                ]
            },
            'ideation': {
                'keywords': ['brainstorm', 'idea', 'suggest', 'propose', 'concept',
                           'innovation', 'creative', 'think', 'imagine', 'possibilities'],
                'patterns': [r'\bbrainstorm\b', r'\bidea\b', r'\bsuggest\b', r'\bpropose\b',
                           r'\bconcept\b', r'\binnovation\b'],
                'examples': [
                    'Brainstorm ideas for a startup',
                    'Suggest concepts for a marketing campaign',
                    'Propose innovative solutions',
                    'Generate ideas for product features'
                ]
            },
            'design': {
                'keywords': ['design', 'layout', 'interface', 'plan', 'blueprint',
                           'architecture', 'structure', 'framework', 'scheme', 'model'],
                'patterns': [r'\bdesign\b', r'\blayout\b', r'\binterface\b', r'\bplan\b',
                           r'\bblueprint\b', r'\barchitecture\b'],
                'examples': [
                    'Design a user interface',
                    'Plan the architecture of a system',
                    'Create a layout for a website',
                    'Develop a framework for the project'
                ]
            },
            'dialogue': {
                'keywords': ['dialogue', 'conversation', 'script', 'chat', 'exchange',
                           'discussion', 'talk', 'speak', 'characters', 'voices'],
                'patterns': [r'\bdialogue\b', r'\bconversation\b', r'\bscript\b',
                           r'\bchat\b', r'\bexchange\b'],
                'examples': [
                    'Write dialogue between two characters',
                    'Create a conversation script',
                    'Develop a chat exchange',
                    'Script a discussion between experts'
                ]
            },
            'general_creative': {
                'keywords': ['creative', 'imagination', 'original', 'unique', 'novel',
                           'innovative', 'artistic', 'expressive', 'inventive', 'inspired'],
                'patterns': [r'\bcreative\b', r'\bimagination\b', r'\boriginal\b',
                           r'\bunique\b', r'\binnovative\b'],
                'examples': [
                    'Be creative with this task',
                    'Use your imagination',
                    'Think of something original',
                    'Create something unique'
                ]
            }
        }
        
        self._cache = {}
        for task, profile in self.task_profiles.items():
            examples = profile['examples']
            self._cache[task] = self.encoder.encode(examples, convert_to_tensor=False)
    
    def classify_task(self, query: str) -> Tuple[str, float, bool, Dict[str, float]]:
        query_lower = query.lower()
        all_scores = {}
        
        for task, profile in self.task_profiles.items():
            score = 0.0
            
            query_embedding = self.encoder.encode([query], convert_to_tensor=False)[0]
            example_embeddings = self._cache[task]
            similarities = np.array([
                np.dot(query_embedding, ex_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(ex_emb))
                for ex_emb in example_embeddings
            ])
            semantic_score = float(np.max(similarities))
            score += semantic_score * 0.6
            
            keyword_matches = sum(1 for kw in profile['keywords'] if kw in query_lower)
            keyword_score = min(keyword_matches / 3.0, 1.0)
            score += keyword_score * 0.25
            
            import re
            pattern_matches = sum(1 for pat in profile['patterns'] if re.search(pat, query_lower))
            pattern_score = min(pattern_matches / 2.0, 1.0)
            score += pattern_score * 0.15
            
            all_scores[task] = score
        
        sorted_tasks = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_task, best_score = sorted_tasks[0]
        second_score = sorted_tasks[1][1] if len(sorted_tasks) > 1 else 0.0
        
        is_ambiguous = (best_score - second_score) < 0.15
        
        alternatives = {task: score for task, score in sorted_tasks[1:4]}
        
        return best_task, best_score, is_ambiguous, alternatives
