import re
import ast
from typing import Dict, Optional, List, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np


class LanguageDetector:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.language_profiles = {
            'python': {
                'exemplars': [
                    "def calculate(x): return x + 1",
                    "import numpy as np\nclass DataProcessor:\n    def __init__(self): pass",
                    "for item in items:\n    if item > 0:\n        result.append(item)",
                    "with open('file.txt') as f:\n    data = f.read()",
                    "lambda x: x ** 2"
                ],
                'ast_parser': self._try_parse_python
            },
            'javascript': {
                'exemplars': [
                    "function calculate(x) { return x + 1; }",
                    "const data = await fetch(url);",
                    "const result = array.map(x => x * 2);",
                    "class Component extends React.Component { }",
                    "export default function App() { }"
                ],
                'ast_parser': None
            },
            'typescript': {
                'exemplars': [
                    "interface User { name: string; age: number; }",
                    "type Result<T> = T | null;",
                    "function process<T>(data: T): T { return data; }",
                    "const value: number = 42;",
                    "class Service implements IService { }"
                ],
                'ast_parser': None
            },
            'java': {
                'exemplars': [
                    "public class Main { public static void main(String[] args) { } }",
                    "private int calculate(int x) { return x + 1; }",
                    "List<String> items = new ArrayList<>();",
                    "@Override public String toString() { return \"\"; }",
                    "public interface Service extends BaseService { }"
                ],
                'ast_parser': None
            },
            'cpp': {
                'exemplars': [
                    "#include <iostream>\nint main() { std::cout << \"Hello\"; }",
                    "template<typename T> class Vector { };",
                    "std::vector<int> data = {1, 2, 3};",
                    "void process(const std::string& input) { }",
                    "namespace app { class Service { }; }"
                ],
                'ast_parser': None
            },
            'go': {
                'exemplars': [
                    "package main\nfunc main() { fmt.Println(\"Hello\") }",
                    "type User struct { Name string; Age int }",
                    "func (u *User) Process() error { return nil }",
                    "go func() { /* async */ }()",
                    "defer file.Close()"
                ],
                'ast_parser': None
            },
            'rust': {
                'exemplars': [
                    "fn main() { println!(\"Hello\"); }",
                    "let mut data: Vec<i32> = vec![1, 2, 3];",
                    "impl MyTrait for MyStruct { }",
                    "pub struct Config { pub name: String }",
                    "match result { Ok(v) => v, Err(e) => panic!() }"
                ],
                'ast_parser': None
            }
        }
        
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self.language_embeddings = {}
        for lang, profile in self.language_profiles.items():
            embeddings = self.encoder.encode(profile['exemplars'], convert_to_tensor=False)
            self.language_embeddings[lang] = embeddings
    
    def detect(self, query: str, code: str = "") -> Dict[str, any]:
        if not code.strip() and not query.strip():
            return {'language': 'unknown', 'confidence': 0.0, 'alternatives': []}
        
        scores = {}
        confidences = {}
        
        query_lower = query.lower()
        explicit_lang = self._extract_explicit_language(query_lower)
        
        if code.strip():
            scores = self._analyze_code(code)
        else:
            scores = {lang: 0.1 for lang in self.language_profiles.keys()}
        
        if explicit_lang and explicit_lang in scores:
            scores[explicit_lang] = max(scores[explicit_lang], 0.75)
        
        semantic_scores = self._semantic_analysis(query, code)
        for lang, sem_score in semantic_scores.items():
            scores[lang] = scores.get(lang, 0.0) * 0.4 + sem_score * 0.6
        
        if not scores or max(scores.values()) < 0.15:
            return {'language': 'unknown', 'confidence': 0.0, 'alternatives': []}
        
        primary = max(scores, key=scores.get)
        confidence = min(scores[primary], 1.0)
        
        alternatives = sorted(
            [(k, v) for k, v in scores.items() if k != primary and v > 0.2],
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        return {
            'language': primary,
            'confidence': confidence,
            'alternatives': alternatives,
            'method': 'ml_hybrid'
        }
    
    def _extract_explicit_language(self, query: str) -> Optional[str]:
        lang_mentions = {
            'python': ['python', 'py', '.py'],
            'javascript': ['javascript', 'js', '.js', 'node'],
            'typescript': ['typescript', 'ts', '.ts'],
            'java': ['java', '.java'],
            'cpp': ['c++', 'cpp', '.cpp', '.cc', '.cxx'],
            'go': ['golang', 'go', '.go'],
            'rust': ['rust', '.rs']
        }
        
        for lang, patterns in lang_mentions.items():
            for pattern in patterns:
                if f' {pattern} ' in f' {query} ' or query.startswith(pattern + ' ') or query.endswith(' ' + pattern):
                    return lang
        return None
    
    def _analyze_code(self, code: str) -> Dict[str, float]:
        scores = {}
        
        for lang, profile in self.language_profiles.items():
            score = 0.0
            
            if profile['ast_parser']:
                ast_result = profile['ast_parser'](code)
                if ast_result['valid']:
                    score = ast_result['confidence']
            else:
                score = self._heuristic_score(code, lang)
            
            scores[lang] = score
        
        return scores
    
    def _try_parse_python(self, code: str) -> Dict[str, any]:
        try:
            ast.parse(code)
            return {'valid': True, 'confidence': 0.95}
        except SyntaxError:
            return {'valid': False, 'confidence': 0.0}
        except Exception:
            return {'valid': False, 'confidence': 0.0}
    
    def _heuristic_score(self, code: str, lang: str) -> float:
        patterns = {
            'python': [
                (r'\bdef\s+\w+\s*\([^)]*\)\s*:', 0.25),
                (r'\bclass\s+\w+', 0.20),
                (r'\bimport\s+\w+', 0.15),
                (r'^\s{4}|\t', 0.10),
                (r'\bfor\s+\w+\s+in\s+', 0.15),
                (r'\bif\s+.*:\s*$', 0.10)
            ],
            'javascript': [
                (r'\bfunction\s+\w+\s*\([^)]*\)\s*\{', 0.25),
                (r'\b(const|let|var)\s+\w+\s*=', 0.20),
                (r'=>', 0.15),
                (r'\.(map|filter|reduce)\(', 0.15),
                (r'async\s+function|await\s+', 0.15)
            ],
            'typescript': [
                (r'\binterface\s+\w+\s*\{', 0.30),
                (r':\s*(string|number|boolean)', 0.25),
                (r'\btype\s+\w+\s*=', 0.20),
                (r'<[A-Z]\w*>', 0.15)
            ],
            'java': [
                (r'\b(public|private|protected)\s+class\s+\w+', 0.30),
                (r'\b(public|private)\s+\w+\s+\w+\s*\([^)]*\)', 0.20),
                (r'System\.out\.print', 0.15),
                (r'\bnew\s+\w+\s*\(', 0.15),
                (r'@\w+', 0.10)
            ],
            'cpp': [
                (r'#include\s*<[^>]+>', 0.30),
                (r'std::\w+', 0.25),
                (r'\btemplate\s*<', 0.20),
                (r'::', 0.10)
            ],
            'go': [
                (r'\bpackage\s+\w+', 0.30),
                (r'\bfunc\s+\w+\s*\([^)]*\)', 0.25),
                (r':=', 0.20),
                (r'\bdefer\s+', 0.15)
            ],
            'rust': [
                (r'\bfn\s+\w+\s*\([^)]*\)', 0.30),
                (r'\blet\s+mut\s+', 0.20),
                (r'\bimpl\s+\w+', 0.20),
                (r'println!\(', 0.15)
            ]
        }
        
        if lang not in patterns:
            return 0.0
        
        score = 0.0
        for pattern, weight in patterns[lang]:
            if re.search(pattern, code, re.MULTILINE):
                score += weight
        
        return min(score, 1.0)
    
    def _semantic_analysis(self, query: str, code: str) -> Dict[str, float]:
        text = f"{query} {code}".strip()
        if not text:
            return {lang: 0.0 for lang in self.language_profiles.keys()}
        
        text_embedding = self.encoder.encode(text, convert_to_tensor=False)
        
        scores = {}
        for lang, exemplar_embeddings in self.language_embeddings.items():
            similarities = []
            for exemplar_emb in exemplar_embeddings:
                sim = np.dot(text_embedding, exemplar_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(exemplar_emb) + 1e-9
                )
                similarities.append(sim)
            
            max_sim = float(np.max(similarities))
            avg_top3 = float(np.mean(sorted(similarities, reverse=True)[:3]))
            scores[lang] = 0.7 * max_sim + 0.3 * avg_top3
        
        return scores
