import re
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer, util


class LanguageDetector:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.patterns = {
            'python': {
                'keywords': ['def', 'import', 'class', '__init__', 'self', 'elif', 'lambda'],
                'symbols': [':', 'self.'],
                'regex': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bclass\s+\w+']
            },
            'javascript': {
                'keywords': ['function', 'const', 'let', 'var', 'async', 'await', 'arrow'],
                'symbols': ['=>', '===', '!=='],
                'regex': [r'\bfunction\s+\w+\s*\(', r'\b(const|let|var)\s+\w+\s*=', r'=>']
            },
            'java': {
                'keywords': ['public', 'private', 'class', 'void', 'static', 'extends'],
                'symbols': ['System.out'],
                'regex': [r'\b(public|private)\s+(class|void|static)', r'System\.out\.print']
            },
            'cpp': {
                'keywords': ['include', 'iostream', 'namespace', 'cout', 'cin'],
                'symbols': ['#include', '::'],
                'regex': [r'#include\s*<', r'std::', r'cout\s*<<']
            },
            'typescript': {
                'keywords': ['interface', 'type', 'readonly', 'implements'],
                'symbols': [': string', ': number'],
                'regex': [r':\s*(string|number|boolean)', r'\binterface\s+\w+']
            },
            'go': {
                'keywords': ['func', 'package', 'import', 'goroutine', 'chan'],
                'symbols': [':=', 'go '],
                'regex': [r'\bfunc\s+\w+\s*\(', r'\bpackage\s+\w+']
            },
            'rust': {
                'keywords': ['fn', 'let', 'mut', 'impl', 'trait'],
                'symbols': ['->'],
                'regex': [r'\bfn\s+\w+\s*\(', r'\blet\s+mut\s+']
            }
        }
        
        self.exemplars = {
            'python': "def calculate(x): import numpy as np; return x + 1",
            'javascript': "function calculate(x) { const result = x + 1; return result; }",
            'java': "public class Calculator { public static void main(String[] args) { System.out.println(); } }",
            'cpp': "#include <iostream> using namespace std; int main() { cout << \"Hello\"; }",
            'typescript': "interface User { name: string; age: number; }",
            'go': "package main import \"fmt\" func main() { fmt.Println() }",
            'rust': "fn main() { let mut x = 5; println!(\"{}\", x); }"
        }
        
        self.exemplar_embeddings = {
            lang: self.encoder.encode(code, convert_to_tensor=True)
            for lang, code in self.exemplars.items()
        }
    
    def detect(self, query: str, code: str = "") -> Dict[str, any]:
        combined = f"{query} {code}".lower()
        scores = {}
        
        for lang, config in self.patterns.items():
            score = 0.0
            
            # 1. Explicit mention in query (weight: 0.4)
            if lang in query.lower() or f".{lang[:2]}" in query.lower():
                score += 0.4
            
            # 2. Keyword matching (weight: 0.3)
            keyword_hits = sum(1 for kw in config['keywords'] if kw in combined)
            score += min(keyword_hits * 0.05, 0.3)
            
            # 3. Symbol presence (weight: 0.15)
            symbol_hits = sum(1 for sym in config['symbols'] if sym in combined)
            score += min(symbol_hits * 0.05, 0.15)
            
            # 4. Regex pattern matching (weight: 0.15)
            if code:
                regex_hits = sum(1 for pattern in config['regex'] if re.search(pattern, code))
                score += min(regex_hits * 0.05, 0.15)
            
            scores[lang] = score
        
        # 5. Semantic similarity for code snippets (weight adjustment)
        if code and len(code) > 20:
            code_embedding = self.encoder.encode(code, convert_to_tensor=True)
            for lang, exemplar_emb in self.exemplar_embeddings.items():
                similarity = float(util.cos_sim(code_embedding, exemplar_emb)[0][0])
                scores[lang] += similarity * 0.2
        
        if not scores or max(scores.values()) == 0:
            return {'language': 'unknown', 'confidence': 0.0, 'alternatives': []}
        
        primary = max(scores, key=scores.get)
        confidence = min(scores[primary], 1.0)
        
        alternatives = sorted(
            [(k, v) for k, v in scores.items() if k != primary and v > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        return {
            'language': primary,
            'confidence': confidence,
            'alternatives': alternatives
        }
