import re
from typing import Dict, List, Optional
from collections import Counter
import math


class CreativeEvaluator:
    
    FIGURATIVE_PATTERNS = {
        'metaphor': [
            r'\bis\s+(?:a|an|the)\s+\w+\s+of\b',
            r'\blike\s+a\b',
            r'\bas\s+\w+\s+as\b'
        ],
        'alliteration': [
            r'\b(\w)\w*\s+\1\w*\s+\1\w*\b'
        ],
        'repetition': [
            r'\b(\w{4,})\b.*\b\1\b'
        ],
        'rhyme_structure': [
            r'(\w+ing)\s+.*\s+(\w+ing)\s*$',
            r'(\w+ed)\s+.*\s+(\w+ed)\s*$'
        ]
    }
    
    TASK_THRESHOLDS = {
        'poem': {'vocab_diversity': 0.6, 'sentence_variety': 0.5, 'figurative_density': 0.15},
        'story': {'vocab_diversity': 0.5, 'sentence_variety': 0.4, 'figurative_density': 0.08},
        'slogan': {'vocab_diversity': 0.7, 'sentence_variety': 0.3, 'figurative_density': 0.2},
        'dialogue': {'vocab_diversity': 0.45, 'sentence_variety': 0.6, 'figurative_density': 0.05},
        'description': {'vocab_diversity': 0.55, 'sentence_variety': 0.4, 'figurative_density': 0.1},
        'default': {'vocab_diversity': 0.5, 'sentence_variety': 0.4, 'figurative_density': 0.1}
    }
    
    def __init__(self):
        pass
    
    def evaluate(self, text: str, task_type: str = 'default', context: str = '') -> Dict:
        thresholds = self.TASK_THRESHOLDS.get(task_type.lower(), self.TASK_THRESHOLDS['default'])
        
        vocab_diversity = self._calculate_vocab_diversity(text)
        sentence_variety = self._calculate_sentence_variety(text)
        figurative_density = self._calculate_figurative_density(text)
        coherence = self._evaluate_coherence(text, context)
        
        vocab_score = min(vocab_diversity / thresholds['vocab_diversity'], 1.0)
        sentence_score = min(sentence_variety / thresholds['sentence_variety'], 1.0)
        figurative_score = min(figurative_density / thresholds['figurative_density'], 1.0)
        
        overall_score = (vocab_score * 0.3 + sentence_score * 0.25 + 
                        figurative_score * 0.25 + coherence * 0.2)
        
        return {
            'overall_score': overall_score,
            'vocab_diversity': vocab_diversity,
            'sentence_variety': sentence_variety,
            'figurative_density': figurative_density,
            'coherence': coherence,
            'meets_threshold': overall_score >= 0.65,
            'breakdown': {
                'vocab_score': vocab_score,
                'sentence_score': sentence_score,
                'figurative_score': figurative_score
            }
        }
    
    def _calculate_vocab_diversity(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 10:
            return 0.0
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
        content_words = [w for w in words if w not in stop_words]
        
        if not content_words:
            return ttr
        
        content_unique = set(content_words)
        content_ttr = len(content_unique) / len(content_words)
        
        return (ttr * 0.4 + content_ttr * 0.6)
    
    def _calculate_sentence_variety(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        lengths = [len(s.split()) for s in sentences]
        
        if not lengths:
            return 0.0
        
        length_variance = sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths)
        length_score = min(math.sqrt(length_variance) / 10.0, 1.0)
        
        structure_patterns = []
        for sent in sentences:
            if sent.strip().endswith('?'):
                structure_patterns.append('question')
            elif sent.strip().endswith('!'):
                structure_patterns.append('exclamation')
            elif ',' in sent:
                structure_patterns.append('complex')
            else:
                structure_patterns.append('simple')
        
        structure_diversity = len(set(structure_patterns)) / 4.0
        
        return (length_score * 0.6 + structure_diversity * 0.4)
    
    def _calculate_figurative_density(self, text: str) -> float:
        if not text:
            return 0.0
        
        total_matches = 0
        
        for category, patterns in self.FIGURATIVE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                total_matches += len(matches)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        density = total_matches / len(sentences)
        
        return min(density, 1.0)
    
    def _evaluate_coherence(self, text: str, context: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8
        
        transition_words = {'however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                           'consequently', 'meanwhile', 'then', 'next', 'finally', 'first', 
                           'second', 'also', 'thus', 'hence', 'besides'}
        
        transitions_used = sum(1 for sent in sentences 
                              if any(tw in sent.lower() for tw in transition_words))
        transition_score = min(transitions_used / max(len(sentences) - 1, 1), 1.0)
        
        words_in_sentences = [set(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
        
        overlap_scores = []
        for i in range(len(words_in_sentences) - 1):
            overlap = len(words_in_sentences[i] & words_in_sentences[i+1])
            total = len(words_in_sentences[i] | words_in_sentences[i+1])
            if total > 0:
                overlap_scores.append(overlap / total)
        
        lexical_cohesion = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.5
        
        return (transition_score * 0.4 + lexical_cohesion * 0.6)
