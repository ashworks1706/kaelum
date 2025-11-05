import re
from typing import Dict, List
from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np


class RepetitionDetector:
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.stylistic_patterns = {
            'anaphora': r'^(\w+(?:\s+\w+){0,2})\s+.*\n(?:.*\n)*?\1\s+',
            'epistrophe': r'(\w+(?:\s+\w+){0,2})\.?\s*$.*\n(?:.*\n)*?.*\1\.?\s*$',
            'refrain': r'^(.{10,50})$.*\n(?:.*\n){2,}?\1$',
            'alliteration': r'\b(\w)\w*\s+\1\w*\s+\1\w*\b'
        }
    
    def detect(self, text: str, context: str = '') -> Dict:
        intentional_score = self._detect_stylistic_patterns(text)
        semantic_repetition = self._detect_semantic_repetition(text)
        lexical_repetition = self._analyze_lexical_repetition(text)
        phrase_repetition = self._analyze_phrase_repetition(text)
        
        redundancy_score = (
            semantic_repetition['score'] * 0.40 +
            lexical_repetition['score'] * 0.30 +
            phrase_repetition['score'] * 0.30
        )
        
        if intentional_score > 0.4:
            is_intentional = True
            quality = 'stylistic'
        elif redundancy_score > 0.7:
            is_intentional = False
            quality = 'poor'
        elif intentional_score > 0.2 and redundancy_score < 0.5:
            is_intentional = True
            quality = 'mixed'
        else:
            is_intentional = False
            quality = 'acceptable'
        
        return {
            'is_intentional': is_intentional,
            'quality': quality,
            'intentional_score': intentional_score,
            'redundancy_score': redundancy_score,
            'semantic_repetition': semantic_repetition,
            'lexical_repetition': lexical_repetition,
            'phrase_repetition': phrase_repetition,
            'patterns_found': self._get_matched_patterns(text)
        }
    
    def _detect_stylistic_patterns(self, text: str) -> float:
        matches = 0
        total_patterns = len(self.stylistic_patterns)
        
        for pattern_name, pattern in self.stylistic_patterns.items():
            found = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if found:
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0
    
    def _detect_semantic_repetition(self, text: str) -> Dict:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return {'score': 0.0, 'repeated_pairs': []}
        
        embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
        
        repetition_pairs = []
        similarity_scores = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                )
                
                if sim > 0.85:
                    repetition_pairs.append((i, j, sim))
                    similarity_scores.append(sim)
        
        if not similarity_scores:
            return {'score': 0.0, 'repeated_pairs': []}
        
        repetition_ratio = len(repetition_pairs) / (len(sentences) * (len(sentences) - 1) / 2)
        avg_similarity = np.mean(similarity_scores)
        
        score = 0.6 * repetition_ratio + 0.4 * (avg_similarity - 0.85) / 0.15
        
        return {
            'score': min(score, 1.0),
            'repeated_pairs': repetition_pairs[:3],
            'count': len(repetition_pairs)
        }
    
    def _analyze_lexical_repetition(self, text: str) -> Dict:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        if len(words) < 10:
            return {'score': 0.0, 'repeated_words': []}
        
        word_counts = Counter(words)
        
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their', 'there',
            'would', 'could', 'should', 'which', 'about', 'after', 'before', 'being',
            'also', 'very', 'more', 'most', 'some', 'such', 'then', 'than', 'them',
            'into', 'only', 'other', 'when', 'where', 'while', 'will', 'your'
        }
        
        content_word_counts = {w: c for w, c in word_counts.items() 
                              if w not in stop_words and c > 1}
        
        if not content_word_counts:
            return {'score': 0.0, 'repeated_words': []}
        
        total_words = len(words)
        unique_words = len(set(words) - stop_words)
        
        lexical_diversity = unique_words / total_words if total_words > 0 else 1.0
        
        max_repetition = max(content_word_counts.values())
        repetition_ratio = sum(content_word_counts.values()) / total_words
        
        score = (1.0 - lexical_diversity) * 0.5 + repetition_ratio * 0.5
        
        repeated_words = [w for w, c in content_word_counts.most_common(5)]
        
        return {
            'score': min(score, 1.0),
            'repeated_words': repeated_words,
            'max_count': max_repetition,
            'lexical_diversity': lexical_diversity
        }
    
    def _analyze_phrase_repetition(self, text: str) -> Dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return {'score': 0.0, 'repeated_phrases': []}
        
        ngrams = []
        for sent in sentences:
            words = sent.split()
            for n in range(3, min(7, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngrams.append(' '.join(words[i:i+n]))
        
        if not ngrams:
            return {'score': 0.0, 'repeated_phrases': []}
        
        ngram_counts = Counter(ngrams)
        repeated_phrases = [(phrase, count) for phrase, count in ngram_counts.items() if count > 1]
        
        if not repeated_phrases:
            return {'score': 0.0, 'repeated_phrases': []}
        
        total_ngrams = len(ngrams)
        repeated_count = sum(count for _, count in repeated_phrases)
        
        score = repeated_count / total_ngrams if total_ngrams > 0 else 0.0
        
        return {
            'score': min(score, 1.0),
            'repeated_phrases': [phrase for phrase, _ in sorted(repeated_phrases, key=lambda x: x[1], reverse=True)[:3]],
            'count': len(repeated_phrases)
        }
    
    def _get_matched_patterns(self, text: str) -> List[str]:
        matched = []
        
        for pattern_name, pattern in self.stylistic_patterns.items():
            found = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if found:
                matched.append(pattern_name)
        
        return matched
