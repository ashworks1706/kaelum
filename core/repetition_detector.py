import re
from typing import Dict, List
from collections import Counter


class RepetitionDetector:
    
    INTENTIONAL_PATTERNS = {
        'anaphora': r'^(\w+(?:\s+\w+){0,2})\s+.*\n(?:.*\n)*?\1\s+',
        'epistrophe': r'(\w+(?:\s+\w+){0,2})\.?\s*$.*\n(?:.*\n)*?.*\1\.?\s*$',
        'refrain': r'^(.{10,50})$.*\n(?:.*\n){2,}?\1$',
        'alliteration': r'\b(\w)\w*\s+\1\w*\s+\1\w*\b'
    }
    
    def __init__(self):
        pass
    
    def detect(self, text: str, context: str = '') -> Dict:
        intentional_score = self._detect_intentional_patterns(text)
        
        word_repetition = self._analyze_word_repetition(text)
        phrase_repetition = self._analyze_phrase_repetition(text)
        
        if intentional_score > 0.4:
            is_intentional = True
            quality = 'stylistic'
        elif word_repetition['score'] > 0.8 or phrase_repetition['score'] > 0.7:
            is_intentional = False
            quality = 'poor'
        elif intentional_score > 0.2:
            is_intentional = True
            quality = 'mixed'
        else:
            is_intentional = False
            quality = 'acceptable'
        
        return {
            'is_intentional': is_intentional,
            'quality': quality,
            'intentional_score': intentional_score,
            'word_repetition': word_repetition,
            'phrase_repetition': phrase_repetition,
            'patterns_found': self._get_matched_patterns(text)
        }
    
    def _detect_intentional_patterns(self, text: str) -> float:
        matches = 0
        total_patterns = len(self.INTENTIONAL_PATTERNS)
        
        for pattern_name, pattern in self.INTENTIONAL_PATTERNS.items():
            found = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if found:
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0
    
    def _analyze_word_repetition(self, text: str) -> Dict:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        if len(words) < 10:
            return {'score': 0.0, 'repeated_words': []}
        
        word_counts = Counter(words)
        
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their', 'there', 
                     'would', 'could', 'should', 'which', 'about', 'after', 'before', 'being'}
        
        content_word_counts = {w: c for w, c in word_counts.items() 
                              if w not in stop_words and c > 1}
        
        if not content_word_counts:
            return {'score': 0.0, 'repeated_words': []}
        
        avg_repetition = sum(content_word_counts.values()) / len(content_word_counts)
        max_repetition = max(content_word_counts.values())
        
        repetition_ratio = max_repetition / len(words)
        
        score = min((avg_repetition - 1) / 3.0 + repetition_ratio, 1.0)
        
        repeated_words = [w for w, c in content_word_counts.most_common(5)]
        
        return {
            'score': score,
            'repeated_words': repeated_words,
            'max_count': max_repetition,
            'avg_count': avg_repetition
        }
    
    def _analyze_phrase_repetition(self, text: str) -> Dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return {'score': 0.0, 'repeated_phrases': []}
        
        ngrams = []
        for sent in sentences:
            words = sent.split()
            for n in range(3, 6):
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
            'score': score,
            'repeated_phrases': [phrase for phrase, _ in sorted(repeated_phrases, key=lambda x: x[1], reverse=True)[:3]],
            'count': len(repeated_phrases)
        }
    
    def _get_matched_patterns(self, text: str) -> List[str]:
        matched = []
        
        for pattern_name, pattern in self.INTENTIONAL_PATTERNS.items():
            found = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if found:
                matched.append(pattern_name)
        
        return matched
