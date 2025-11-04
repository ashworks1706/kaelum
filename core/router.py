import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("kaelum.router")
logger.setLevel(logging.INFO)


class QueryType(Enum):
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class ReasoningStrategy(Enum):
    SYMBOLIC_HEAVY = "symbolic_heavy"
    FACTUAL_HEAVY = "factual_heavy"
    BALANCED = "balanced"
    FAST = "fast"
    DEEP = "deep"


@dataclass
class RoutingDecision:
    query_type: QueryType
    worker_specialty: str
    confidence: float
    reasoning: str = ""
    secondary_types: List[QueryType] = None
    complexity_score: float = 0.0
    use_tree_cache: bool = True
    max_tree_depth: int = 5
    num_simulations: int = 10
    
    def __post_init__(self):
        if self.secondary_types is None:
            self.secondary_types = []


@dataclass
class RoutingOutcome:
    query: str
    query_type: QueryType
    strategy: ReasoningStrategy
    decision: RoutingDecision
    success: bool
    accuracy_score: float
    latency_ms: float
    cost: float
    symbolic_passed: bool
    factual_passed: bool
    reflection_iterations: int
    timestamp: float


@dataclass
class NeuralRoutingFeatures:
    embedding: np.ndarray
    query_length: int
    word_count: int
    has_numbers: bool
    has_math_symbols: bool
    has_code_keywords: bool
    has_logic_keywords: bool
    question_words: int
    complexity_score: float
    avg_word_length: float
    uppercase_ratio: float
    punctuation_count: int
    has_parentheses: bool
    has_quotes: bool
    
    def to_tensor(self):
        handcrafted = np.array([
            self.query_length / 200.0,
            self.word_count / 50.0,
            float(self.has_numbers),
            float(self.has_math_symbols),
            float(self.has_code_keywords),
            float(self.has_logic_keywords),
            self.question_words / 5.0,
            self.complexity_score,
            self.avg_word_length / 10.0,
            self.uppercase_ratio,
            self.punctuation_count / 10.0,
            float(self.has_parentheses),
            float(self.has_quotes),
            0.0
        ], dtype=np.float32)
        
        features = np.concatenate([self.embedding, handcrafted])
        return torch.from_numpy(features).float()


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 398, hidden_dim: int = 256):
        super().__init__()
        
        self.embed_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.worker_head = nn.Linear(hidden_dim, 6)
        self.depth_head = nn.Linear(hidden_dim, 1)
        self.sims_head = nn.Linear(hidden_dim, 1)
        self.cache_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.embed_proj(x)
        h = h + self.res1(h)
        h = h + self.res2(h)
        
        return {
            'worker_logits': self.worker_head(h),
            'depth_logits': self.depth_head(h),
            'sims_logits': self.sims_head(h),
            'cache_logits': self.cache_head(h)
        }


class Router:
    def __init__(self, learning_enabled: bool = True, data_dir: str = ".kaelum/routing", 
                 model_path: Optional[str] = None, device: str = "cpu"):
        self.learning_enabled = learning_enabled
        self.device = device
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.model_file = self.data_dir / "model.pt"
        self.outcomes = []
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.policy_network = PolicyNetwork().to(device)
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        elif self.model_file.exists():
            self._load_model(str(self.model_file))
        
        self.idx_to_worker = {
            0: "math",
            1: "logic",
            2: "code",
            3: "factual",
            4: "creative",
            5: "analysis"
        }
        
    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        start_time = time.time()
        
        features = self._extract_features(query, context)
        
        self.policy_network.eval()
        with torch.no_grad():
            feature_tensor = features.to_tensor().unsqueeze(0).to(self.device)
            outputs = self.policy_network(feature_tensor)
            
            worker_probs = torch.softmax(outputs['worker_logits'], dim=-1)
            worker_idx = torch.argmax(worker_probs, dim=-1).item()
            confidence = worker_probs[0, worker_idx].item()
            
            max_tree_depth = int(torch.clamp(outputs['depth_logits'], 3, 10).item())
            num_simulations = int(torch.clamp(outputs['sims_logits'], 5, 25).item())
            use_cache = torch.sigmoid(outputs['cache_logits']).item() > 0.5
        
        worker_specialty = self.idx_to_worker[worker_idx]
        query_type = self._classify_query_type(query)
        
        reasoning = f"Neural router: {worker_specialty} (conf={confidence:.2f}, comp={features.complexity_score:.2f})"
        
        routing_time = (time.time() - start_time) * 1000
        logger.info(f"Route: {worker_specialty} | conf={confidence:.2f} | depth={max_tree_depth} | sims={num_simulations} | {routing_time:.1f}ms")
        
        return RoutingDecision(
            query_type=query_type,
            worker_specialty=worker_specialty,
            confidence=confidence,
            reasoning=reasoning,
            secondary_types=[],
            complexity_score=features.complexity_score,
            max_tree_depth=max_tree_depth,
            num_simulations=num_simulations,
            use_tree_cache=use_cache
        )
    
    def record_outcome(self, decision: RoutingDecision, result: Dict[str, Any]):
        if not self.learning_enabled:
            return
        
        outcome = RoutingOutcome(
            query=result.get("query", ""),
            query_type=decision.query_type,
            strategy=ReasoningStrategy.BALANCED,
            decision=decision,
            success=result.get("success", False),
            accuracy_score=result.get("confidence", 0.0),
            latency_ms=result.get("execution_time", 0) * 1000,
            cost=result.get("cost", 0.0),
            symbolic_passed=result.get("verification_passed", False),
            factual_passed=result.get("verification_passed", False),
            reflection_iterations=0,
            timestamp=time.time()
        )
        
        self.outcomes.append(outcome)
        
        if len(self.outcomes) % 10 == 0:
            self._save_training_data()
    
    def _extract_features(self, query: str, context: Optional[Dict] = None) -> NeuralRoutingFeatures:
        embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        q_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        has_code = any(x in q_lower for x in ["code", "function", "class", "debug", "implement"])
        has_math = any(x in q_lower for x in ["calculate", "solve", "equation", "derivative", "integral"])
        has_logic = any(x in q_lower for x in ["if", "then", "therefore", "prove", "logic"])
        
        num_numbers = sum(c.isdigit() for c in query)
        num_symbols = sum(not c.isalnum() and not c.isspace() for c in query)
        num_upper = sum(c.isupper() for c in query)
        num_punct = sum(c in '.,!?;:' for c in query)
        question_words = sum(q_lower.count(w) for w in ["what", "why", "how", "when", "where", "who"])
        
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / max(word_count, 1)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        complexity = min(1.0, (
            0.2 * (word_count / 50) +
            0.2 * (avg_word_length / 10) +
            0.2 * (1 - lexical_diversity) +
            0.2 * (num_numbers / 20) +
            0.2 * (num_symbols / 20)
        ))
        
        return NeuralRoutingFeatures(
            embedding=embedding,
            query_length=len(query),
            word_count=word_count,
            has_numbers=num_numbers > 0,
            has_math_symbols=any(c in query for c in '+-*/=^'),
            has_code_keywords=has_code,
            has_logic_keywords=has_logic,
            question_words=question_words,
            complexity_score=complexity,
            avg_word_length=avg_word_length,
            uppercase_ratio=num_upper / max(len(query), 1),
            punctuation_count=num_punct,
            has_parentheses='(' in query or ')' in query,
            has_quotes='"' in query or "'" in query
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        q_lower = query.lower()
        
        if any(x in q_lower for x in ["code", "function", "debug", "implement", "algorithm"]):
            return QueryType.CODE
        elif any(x in q_lower for x in ["calculate", "solve", "prove", "equation", "theorem"]):
            return QueryType.MATH
        elif any(x in q_lower for x in ["fact", "history", "definition", "explain", "what is"]):
            return QueryType.FACTUAL
        elif any(x in q_lower for x in ["story", "poem", "creative", "imagine", "write"]):
            return QueryType.CREATIVE
        else:
            return QueryType.UNKNOWN
    
    def _load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["model_state_dict"])
    
    def save_model(self):
        torch.save({
            "model_state_dict": self.policy_network.state_dict(),
            "input_dim": 398,
            "hidden_dim": 256,
        }, self.model_file)
    
    def _save_training_data(self):
        data = [
            {
                "query": o.query,
                "query_type": o.query_type.value,
                "worker": o.decision.worker_specialty,
                "success": o.success,
                "confidence": o.accuracy_score,
                "latency_ms": o.latency_ms,
                "timestamp": o.timestamp
            }
            for o in self.outcomes
        ]
        
        training_file = self.data_dir / "training_data.json"
        with open(training_file, "w") as f:
            json.dump(data, f, indent=2)

