import re
import ast
from typing import Dict, Optional, List, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np


class LanguageDetector:
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
        self.language_profiles = {
            'python': {
                'exemplars': [
                    "def calculate(x): return x + 1",
                    "import numpy as np\nclass DataProcessor:\n    def __init__(self): pass",
                    "for item in items:\n    if item > 0:\n        result.append(item)",
                    "with open('file.txt') as f:\n    data = f.read()",
                    "lambda x: x ** 2",
                    "@decorator\ndef function():\n    pass",
                    "async def fetch_data():\n    await process()"
                ],
                'ast_parser': self._try_parse_python,
                'exclusion_patterns': [
                    r'\bpublic\s+class\b',
                    r'\bprivate\s+void\b',
                    r'System\.out\.print',
                    r'#include\s*<'
                ]
            },
            'javascript': {
                'exemplars': [
                    "function calculate(x) { return x + 1; }",
                    "const data = await fetch(url);",
                    "const result = array.map(x => x * 2);",
                    "class Component extends React.Component { }",
                    "export default function App() { }",
                    "let value = async () => await api.call();"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+',
                    r':\s*$',
                    r'\bfn\s+\w+\s*\('
                ]
            },
            'typescript': {
                'exemplars': [
                    "interface User { name: string; age: number; }",
                    "type Result<T> = T | null;",
                    "function process<T>(data: T): T { return data; }",
                    "const value: number = 42;",
                    "class Service implements IService { }",
                    "enum Status { Active, Inactive }"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+$',
                    r'\bfn\s+\w+\s*\('
                ]
            },
            'java': {
                'exemplars': [
                    "public class Main { public static void main(String[] args) { } }",
                    "private int calculate(int x) { return x + 1; }",
                    "List<String> items = new ArrayList<>();",
                    "@Override public String toString() { return \"\"; }",
                    "public interface Service extends BaseService { }",
                    "import java.util.*;\npublic class Test {}"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+$',
                    r'=>',
                    r'\bfn\s+\w+\s*\('
                ]
            },
            'cpp': {
                'exemplars': [
                    "#include <iostream>\nint main() { std::cout << \"Hello\"; }",
                    "template<typename T> class Vector { };",
                    "std::vector<int> data = {1, 2, 3};",
                    "void process(const std::string& input) { }",
                    "namespace app { class Service { }; }",
                    "#include <vector>\nusing namespace std;"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+$',
                    r'\bfunction\s+\w+\s*\(',
                    r'\bfn\s+\w+\s*\('
                ]
            },
            'go': {
                'exemplars': [
                    "package main\nfunc main() { fmt.Println(\"Hello\") }",
                    "type User struct { Name string; Age int }",
                    "func (u *User) Process() error { return nil }",
                    "go func() { /* async */ }()",
                    "defer file.Close()",
                    "import \"fmt\"\nfunc test() {}"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+$',
                    r'\bfunction\s+\w+\s*\(',
                    r'#include\s*<'
                ]
            },
            'rust': {
                'exemplars': [
                    "fn main() { println!(\"Hello\"); }",
                    "let mut data: Vec<i32> = vec![1, 2, 3];",
                    "impl MyTrait for MyStruct { }",
                    "pub struct Config { pub name: String }",
                    "match result { Ok(v) => v, Err(e) => panic!() }",
                    "use std::collections::HashMap;"
                ],
                'ast_parser': None,
                'exclusion_patterns': [
                    r'\bdef\s+\w+\s*\(',
                    r'^\s*import\s+\w+$',
                    r'\bfunction\s+\w+\s*\(',
                    r'#include\s*<'
                ]
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
            scores[explicit_lang] = min(scores[explicit_lang] + 0.25, 0.95)
        
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
            
            exclusion_patterns = profile.get('exclusion_patterns', [])
            exclusion_penalty = sum(1 for pattern in exclusion_patterns if re.search(pattern, code, re.MULTILINE))
            
            if exclusion_penalty > 2:
                scores[lang] = 0.0
                continue
            
            if profile['ast_parser']:
                ast_result = profile['ast_parser'](code)
                if ast_result['valid']:
                    score = ast_result['confidence']
                    scores[lang] = score * (1.0 - exclusion_penalty * 0.2)
                    continue
            
            heuristic_score = self._heuristic_score(code, lang)
            scores[lang] = max(0.0, heuristic_score - exclusion_penalty * 0.15)
        
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
                (r'\bdef\s+\w+\s*\([^)]*\)\s*:', 0.30, True),
                (r'\bclass\s+\w+', 0.25, True),
                (r'^\s*import\s+\w+', 0.20, True),
                (r'^\s{4}[^\s]|\t[^\s]', 0.12, False),
                (r'\bfor\s+\w+\s+in\s+', 0.18, True),
                (r'\bif\s+.*:\s*$', 0.15, False),
                (r'@\w+\s*\n\s*def', 0.20, True)
            ],
            'javascript': [
                (r'\bfunction\s+\w+\s*\([^)]*\)\s*\{', 0.30, True),
                (r'\b(const|let|var)\s+\w+\s*=', 0.25, True),
                (r'=>', 0.20, True),
                (r'\.(map|filter|reduce|forEach)\(', 0.20, True),
                (r'\basync\s+function|\bawait\s+', 0.18, True),
                (r'\bconsole\.log\(', 0.15, True)
            ],
            'typescript': [
                (r'\binterface\s+\w+\s*\{', 0.35, True),
                (r':\s*(string|number|boolean|any)\b', 0.30, True),
                (r'\btype\s+\w+\s*=', 0.25, True),
                (r'<[A-Z]\w*>', 0.20, False),
                (r'\benum\s+\w+\s*\{', 0.25, True)
            ],
            'java': [
                (r'\b(public|private|protected)\s+class\s+\w+', 0.35, True),
                (r'\b(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)', 0.25, True),
                (r'System\.out\.print', 0.20, True),
                (r'\bnew\s+\w+\s*\(', 0.18, True),
                (r'@\w+\s*\n\s*(public|private)', 0.20, True),
                (r'\bimport\s+java\.', 0.25, True)
            ],
            'cpp': [
                (r'#include\s*<[^>]+>', 0.35, True),
                (r'std::\w+', 0.30, True),
                (r'\btemplate\s*<', 0.25, True),
                (r'::', 0.15, False),
                (r'\bnamespace\s+\w+', 0.20, True),
                (r'using\s+namespace\s+std', 0.25, True)
            ],
            'go': [
                (r'\bpackage\s+\w+', 0.35, True),
                (r'\bfunc\s+\w+\s*\([^)]*\)', 0.30, True),
                (r':=', 0.25, True),
                (r'\bdefer\s+', 0.20, True),
                (r'\bgo\s+func', 0.25, True),
                (r'\btype\s+\w+\s+struct', 0.25, True)
            ],
            'rust': [
                (r'\bfn\s+\w+\s*\([^)]*\)', 0.35, True),
                (r'\blet\s+mut\s+', 0.25, True),
                (r'\bimpl\s+\w+', 0.25, True),
                (r'println!\(', 0.20, True),
                (r'\bmatch\s+\w+\s*\{', 0.20, True),
                (r'\buse\s+std::', 0.25, True)
            ]
        }
        
        if lang not in patterns:
            return 0.0
        
        score = 0.0
        matched_unique = set()
        
        for pattern, weight, is_unique in patterns[lang]:
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                if is_unique:
                    if pattern not in matched_unique:
                        score += weight
                        matched_unique.add(pattern)
                else:
                    count = len(matches)
                    score += weight * min(count / 3.0, 1.0)
        
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
