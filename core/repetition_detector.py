import re
from typing import Dict, List, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class RepetitionDetector:
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.tfidf = TfidfVectorizer(max_features=100, stop_words=None)
        self._calibrated_thresholds = {
            'semantic': 0.85,
            'lexical': 0.7,
            'phrase': 0.6
        }
    
    def detect(self, text: str, context: str = '') -> Dict:
        intentional_score = self._detect_stylistic_patterns_ml(text)
        semantic_repetition = self._detect_semantic_repetition(text)
        lexical_repetition = self._analyze_lexical_repetition(text)
        phrase_repetition = self._analyze_phrase_repetition_semantic(text)
        
        redundancy_score = (
            semantic_repetition['score'] * 0.40 +
            lexical_repetition['score'] * 0.30 +
            phrase_repetition['score'] * 0.30
        )
        
        threshold_intentional = self._adaptive_threshold(text, 'intentional')
        threshold_redundancy = self._adaptive_threshold(text, 'redundancy')
        
        if intentional_score > threshold_intentional:
            is_intentional = True
            quality = 'stylistic'
        elif redundancy_score > threshold_redundancy:
            is_intentional = False
            quality = 'poor'
        elif intentional_score > threshold_intentional * 0.5 and redundancy_score < threshold_redundancy * 0.7:
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
            'phrase_repetition': phrase_repetition
        }
    
    def _adaptive_threshold(self, text: str, threshold_type: str) -> float:
        base_thresholds = {
            'intentional': 0.4,
            'redundancy': 0.7
        }
        base = base_thresholds.get(threshold_type, 0.5)
        
        words = text.split()
        if len(words) < 50:
            adjustment = 0.1
        elif len(words) > 200:
            adjustment = -0.05
        else:
            adjustment = 0.0
        
        return max(0.2, min(0.9, base + adjustment))
    
    def _detect_stylistic_patterns_ml(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        embeddings = self.encoder.encode(sentences, convert_to_tensor=False)
        
        pattern_scores = []
        for i in range(len(sentences) - 1):
            for j in range(i + 1, min(i + 4, len(sentences))):
                if i == j:
                    continue
                    
                words_i = sentences[i].lower().split()
                words_j = sentences[j].lower().split()
                
                if len(words_i) < 3 or len(words_j) < 3:
                    continue
                
                start_match = words_i[:3] == words_j[:3]
                end_match = words_i[-3:] == words_j[-3:]
                
                if start_match or end_match:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                    )
                    pattern_scores.append(sim)
        
        if not pattern_scores:
            return 0.0
        
        return float(np.mean(pattern_scores))
    
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
                
                threshold = self._calibrated_thresholds['semantic']
                if sim > threshold:
                    repetition_pairs.append((i, j, sim))
                    similarity_scores.append(sim)
        
        if not similarity_scores:
            return {'score': 0.0, 'repeated_pairs': []}
        
        repetition_ratio = len(repetition_pairs) / (len(sentences) * (len(sentences) - 1) / 2)
        avg_similarity = np.mean(similarity_scores)
        
        score = 0.6 * repetition_ratio + 0.4 * (avg_similarity - self._calibrated_thresholds['semantic']) / (1.0 - self._calibrated_thresholds['semantic'])
        
        return {
            'score': min(score, 1.0),
            'repeated_pairs': repetition_pairs[:3],
            'count': len(repetition_pairs)
        }
    
    def _analyze_lexical_repetition(self, text: str) -> Dict:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        if len(words) < 10:
            return {'score': 0.0, 'repeated_words': []}
        
        try:
            corpus_sample = [text]
            self.tfidf.fit(corpus_sample)
            word_scores = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))
            
            low_idf_words = {w for w, score in word_scores.items() if score < 2.0}
        except:
            low_idf_words = {
                'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their', 'there',
                'would', 'could', 'should', 'which', 'about', 'after', 'before', 'being'
            }
        
        word_counts = Counter(words)
        content_word_counts = {w: c for w, c in word_counts.items() 
                              if w not in low_idf_words and c > 1}
        
        if not content_word_counts:
            return {'score': 0.0, 'repeated_words': []}
        
        total_words = len(words)
        unique_words = len(set(words) - low_idf_words)
        
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
    
    def _analyze_phrase_repetition_semantic(self, text: str) -> Dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return {'score': 0.0, 'repeated_phrases': []}
        
        all_phrases = []
        phrase_to_text = {}
        
        for sent in sentences:
            words = sent.split()
            for n in range(3, min(7, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    all_phrases.append(phrase)
                    if phrase not in phrase_to_text:
                        phrase_to_text[phrase] = []
                    phrase_to_text[phrase].append(sent)
        
        if len(all_phrases) < 10:
            return {'score': 0.0, 'repeated_phrases': []}
        
        unique_phrases = list(set(all_phrases))
        if len(unique_phrases) < 2:
            return {'score': 0.0, 'repeated_phrases': []}
        
        embeddings = self.encoder.encode(unique_phrases, convert_to_tensor=False)
        
        semantic_clusters = []
        visited = set()
        
        for i in range(len(embeddings)):
            if i in visited:
                continue
            
            cluster = [unique_phrases[i]]
            visited.add(i)
            
            for j in range(i + 1, len(embeddings)):
                if j in visited:
                    continue
                
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                )
                
                if sim > 0.9:
                    cluster.append(unique_phrases[j])
                    visited.add(j)
            
            if len(cluster) > 1:
                semantic_clusters.append(cluster)
        
        if not semantic_clusters:
            return {'score': 0.0, 'repeated_phrases': []}
        
        total_phrase_count = len(all_phrases)
        repeated_count = sum(len(cluster) for cluster in semantic_clusters)
        
        score = repeated_count / total_phrase_count if total_phrase_count > 0 else 0.0
        
        top_clusters = sorted(semantic_clusters, key=lambda x: len(x), reverse=True)[:3]
        repeated_phrases = [cluster[0] for cluster in top_clusters]
        
        return {
            'score': min(score, 1.0),
            'repeated_phrases': repeated_phrases,
            'count': len(semantic_clusters)
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
        return []
