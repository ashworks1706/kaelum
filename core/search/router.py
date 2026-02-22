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
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from core.shared_encoder import get_shared_encoder
from core.paths import DEFAULT_ROUTER_DIR
from core.learning.human_feedback import HumanFeedbackEngine

from ..detectors import DomainClassifier

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
    def __init__(self, learning_enabled: bool = True, data_dir: str = DEFAULT_ROUTER_DIR,
                 model_path: Optional[str] = None, device: str = "cpu", embedding_model: str = "all-MiniLM-L6-v2",
                 buffer_size: int = 32, learning_rate: float = 0.001, exploration_rate: float = 0.1,
                 depth_range: tuple = (3, 10), sims_range: tuple = (5, 25)):
        self.learning_enabled = learning_enabled
        self.device = device
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.model_file = self.data_dir / "model.pt"
        self.outcomes = []
        
        self._load_outcomes()
        
        self.buffer_size = buffer_size
        self.exploration_rate = exploration_rate
        self.training_step_count = 0
        
        self.encoder = get_shared_encoder(embedding_model, device='cpu')
        self.policy_network = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.training_buffer = []
        self.domain_classifier = DomainClassifier(embedding_model=embedding_model)
        
        self.feedback_engine = HumanFeedbackEngine()
        logger.info(f"ROUTER: Human feedback engine initialized")
        self.depth_range = depth_range
        self.sims_range = sims_range
        
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
        
        logger.info("=" * 80)
        logger.info("ROUTER: Starting query analysis")
        logger.info(f"ROUTER: Query = '{query}'")
        
        features = self._extract_features(query, context)
        
        logger.info("ROUTER: Feature extraction complete")
        logger.info(f"  - Word count: {features.word_count}")
        logger.info(f"  - Has numbers: {features.has_numbers}")
        logger.info(f"  - Has math symbols: {features.has_math_symbols}")
        logger.info(f"  - Has code keywords: {features.has_code_keywords}")
        logger.info(f"  - Has logic keywords: {features.has_logic_keywords}")
        logger.info(f"  - Question words: {features.question_words}")
        logger.info(f"  - Complexity score: {features.complexity_score:.3f}")
        logger.info("ROUTER: Running neural network classification")
        self.policy_network.eval()
        with torch.no_grad():
            feature_tensor = features.to_tensor().unsqueeze(0).to(self.device)
            outputs = self.policy_network(feature_tensor)
            
            worker_probs = torch.softmax(outputs['worker_logits'], dim=-1)
            adjusted_probs = worker_probs

            if self.learning_enabled and np.random.random() < self.exploration_rate:
                worker_idx = np.random.randint(0, len(self.idx_to_worker))
                confidence = adjusted_probs[0, worker_idx].item()
                logger.info(f"ROUTER: EXPLORATION MODE - Random worker selected")
            else:
                worker_idx = torch.argmax(adjusted_probs, dim=-1).item()
                confidence = adjusted_probs[0, worker_idx].item()
            
            logger.info("ROUTER: Worker probabilities:")
            for idx, prob in enumerate(adjusted_probs[0]):
                worker_name = self.idx_to_worker[idx]
                orig_prob = worker_probs[0, idx].item()
                logger.info(f"  - {worker_name}: {prob.item():.3f} (original: {orig_prob:.3f})")
            
            # Constrain predictions to reasonable ranges based on empirical observations:
            # Depth 3-10: Shallower wastes MCTS potential, deeper hits diminishing returns
            # Simulations 5-25: Fewer misses good paths, more wastes time for marginal gain
            max_tree_depth = int(torch.clamp(outputs['depth_logits'], 3, 10).item())
            num_simulations = int(torch.clamp(outputs['sims_logits'], 5, 25).item())
            use_cache = torch.sigmoid(outputs['cache_logits']).item() > 0.5
        
        worker_specialty = self.idx_to_worker[worker_idx]
        query_type = self._classify_query_type(query)
        
        reasoning = f"Neural router: {worker_specialty} (conf={confidence:.2f}, comp={features.complexity_score:.2f})"
        
        routing_time = (time.time() - start_time) * 1000
        
        logger.info(f"ROUTER: DECISION - {worker_specialty.upper()}")
        logger.info(f"  - Confidence: {confidence:.3f}")
        logger.info(f"  - Query type: {query_type.value}")
        logger.info(f"  - Max tree depth: {max_tree_depth}")
        logger.info(f"  - Num simulations: {num_simulations}")
        logger.info(f"  - Use cache: {use_cache}")
        logger.info(f"  - Routing time: {routing_time:.1f}ms")
        logger.info(f"  - Reasoning: {reasoning}")
        logger.info("=" * 80)
        
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
        
        features = self._extract_features(result.get("query", ""))
        worker_to_idx = {v: k for k, v in self.idx_to_worker.items()}
        worker_idx = worker_to_idx.get(decision.worker_specialty, 0)
        
        reward = 1.0 if result.get("verification_passed", False) else 0.0
        reward *= result.get("confidence", 0.5)
        
        avg_reward = result.get("avg_reward", reward)
        quality = max(reward, avg_reward)
        
        self.training_buffer.append({
            "features": features,
            "worker_idx": worker_idx,
            "reward": reward,
            "quality": quality,
            "depth": decision.max_tree_depth,
            "sims": decision.num_simulations,
            "use_cache": decision.use_tree_cache
        })
        
        logger.info(f"ROUTER LEARNING: Recorded outcome #{len(self.outcomes)}")
        logger.info(f"  - Worker: {decision.worker_specialty}")
        logger.info(f"  - Success: {result.get('verification_passed', False)}")
        logger.info(f"  - Confidence: {result.get('confidence', 0.0):.3f}")
        logger.info(f"  - Reward: {reward:.3f}")
        logger.info(f"  - Training buffer: {len(self.training_buffer)}/{self.buffer_size}")
        
        self._save_outcome(outcome)
        
        self._save_training_data()
        
        if len(self.training_buffer) >= self.buffer_size:
            logger.info(f"ROUTER TRAINING: Buffer full ({len(self.training_buffer)} samples), starting training...")
            self._train_step()
            self.training_buffer = []
    
    def _train_step(self):
        if len(self.training_buffer) < 8:
            return
        
        high_quality_samples = [item for item in self.training_buffer if item.get("quality", 0.0) > 0.8]
        
        if len(high_quality_samples) < 4:
            logger.info(f"ROUTER TRAINING: Skipped - only {len(high_quality_samples)} high-quality samples (need ≥4)")
            return
        
        logger.info(f"ROUTER TRAINING: Using {len(high_quality_samples)}/{len(self.training_buffer)} high-quality samples (quality > 0.8)")
        
        training_samples = high_quality_samples
        
        self.policy_network.train()
        
        features_list = [item["features"].to_tensor() for item in training_samples]
        features_batch = torch.stack(features_list).to(self.device)
        
        worker_targets = torch.tensor([item["worker_idx"] for item in training_samples], dtype=torch.long).to(self.device)
        rewards = torch.tensor([item["reward"] for item in training_samples], dtype=torch.float32).to(self.device)
        
        outputs = self.policy_network(features_batch)
        
        worker_loss = nn.CrossEntropyLoss()(outputs['worker_logits'], worker_targets)
        worker_loss = worker_loss * rewards.mean()
        
        depth_targets = torch.tensor([item["depth"] for item in training_samples], dtype=torch.float32).unsqueeze(1).to(self.device)
        depth_loss = nn.MSELoss()(outputs['depth_logits'], depth_targets)
        
        sims_targets = torch.tensor([item["sims"] for item in training_samples], dtype=torch.float32).unsqueeze(1).to(self.device)
        sims_loss = nn.MSELoss()(outputs['sims_logits'], sims_targets)
        
        cache_targets = torch.tensor([float(item["use_cache"]) for item in training_samples], dtype=torch.float32).unsqueeze(1).to(self.device)
        cache_loss = nn.BCEWithLogitsLoss()(outputs['cache_logits'], cache_targets)
        
        total_loss = worker_loss + 0.1 * depth_loss + 0.1 * sims_loss + 0.05 * cache_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step_count += 1
        
        logger.info(f"ROUTER TRAINING: Completed training step #{self.training_step_count}")
        logger.info(f"  - Total outcomes: {len(self.outcomes)}")
        logger.info(f"  - Batch size: {len(self.training_buffer)}")
        logger.info(f"  - Loss: {total_loss.item():.4f}")
        logger.info(f"    - Worker loss: {worker_loss.item():.4f}")
        logger.info(f"    - Depth loss: {depth_loss.item():.4f}")
        logger.info(f"    - Sims loss: {sims_loss.item():.4f}")
        logger.info(f"    - Cache loss: {cache_loss.item():.4f}")
        logger.info(f"  - Avg reward: {rewards.mean().item():.3f}")
        
        self.save_model()
        
        if self.training_step_count % 10 == 0:
            old_rate = self.exploration_rate
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
            if old_rate != self.exploration_rate:
                logger.info(f"  - Exploration rate decreased: {old_rate:.3f} → {self.exploration_rate:.3f}")
    
    def _extract_features(self, query: str, context: Optional[Dict] = None) -> NeuralRoutingFeatures:
        import re
        
        embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        words = query.split()
        word_count = len(words)
        
        num_numbers = sum(c.isdigit() for c in query)
        num_symbols = sum(not c.isalnum() and not c.isspace() for c in query)
        num_upper = sum(c.isupper() for c in query)
        num_punct = sum(c in '.,!?;:' for c in query)
        
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / max(word_count, 1)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        question_words_set = {'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom'}
        question_words = sum(1 for w in words if w.lower() in question_words_set)
        
        has_math_symbols = bool(re.search(r'\d+\s*[+\-*/^=]\s*\d+', query)) or \
                          any(sym in query for sym in ['√', '∫', '∂', '∑', '∏', 'derivative', 'integral'])
        
        domain_features = self.domain_classifier.get_domain_features(query)
        has_code_keywords = domain_features['has_code_domain'] > 0.5
        has_logic_keywords = domain_features['has_logic_domain'] > 0.5
        
        structural_complexity = (
            (word_count / 100.0) * 0.3 +
            lexical_diversity * 0.3 +
            (avg_word_length / 15.0) * 0.2 +
            (num_symbols / 30.0) * 0.2
        )
        
        return NeuralRoutingFeatures(
            embedding=embedding,
            query_length=len(query),
            word_count=word_count,
            has_numbers=num_numbers > 2,
            has_math_symbols=has_math_symbols,
            has_code_keywords=has_code_keywords,
            has_logic_keywords=has_logic_keywords,
            question_words=question_words,
            complexity_score=min(structural_complexity, 1.0),
            avg_word_length=avg_word_length,
            uppercase_ratio=num_upper / max(len(query), 1),
            punctuation_count=num_punct,
            has_parentheses='(' in query or ')' in query,
            has_quotes='"' in query or "'" in query
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        worker_to_type = {
            "math": QueryType.MATH,
            "logic": QueryType.LOGIC,
            "code": QueryType.CODE,
            "factual": QueryType.FACTUAL,
            "creative": QueryType.CREATIVE,
            "analysis": QueryType.ANALYSIS
        }
        
        features = self._extract_features(query)
        self.policy_network.eval()
        with torch.no_grad():
            feature_tensor = features.to_tensor().unsqueeze(0).to(self.device)
            outputs = self.policy_network(feature_tensor)
            worker_idx = torch.argmax(outputs['worker_logits'], dim=-1).item()
            worker = self.idx_to_worker[worker_idx]
        
        return worker_to_type.get(worker, QueryType.UNKNOWN)
    
    def _load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "training_step_count" in checkpoint:
            self.training_step_count = checkpoint["training_step_count"]
        if "exploration_rate" in checkpoint:
            self.exploration_rate = checkpoint["exploration_rate"]
        logger.info(f"ROUTER: Model loaded from {path}")
        logger.info(f"  - Training steps: {self.training_step_count}")
        logger.info(f"  - Exploration rate: {self.exploration_rate:.3f}")
    
    def _load_outcomes(self):
        """Load existing routing outcomes from JSONL file."""
        if not self.outcomes_file.exists():
            logger.debug("ROUTER: No existing outcomes file found")
            return
        
        try:
            count = 0
            with open(self.outcomes_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)

                        outcome = RoutingOutcome(
                            query=data['query'],
                            query_type=QueryType(data['query_type']),
                            decision=RoutingDecision(
                                worker_specialty=data['worker'],
                                confidence=data['confidence'],
                                reasoning=data.get('reasoning', ''),
                                max_tree_depth=data.get('max_tree_depth', 4),
                                num_simulations=data.get('num_simulations', 10),
                                use_cache=data.get('use_cache', True),
                                query_type=QueryType(data['query_type'])
                            ),
                            success=data['success'],
                            accuracy_score=data.get('accuracy_score', data.get('confidence', 0.5)),
                            latency_ms=data['latency_ms'],
                            timestamp=data['timestamp']
                        )
                        self.outcomes.append(outcome)
                        count += 1
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"ROUTER: Failed to parse outcome line: {e}")
                        continue
            
            logger.info(f"ROUTER: Loaded {count} existing routing outcomes from {self.outcomes_file}")
        except Exception as e:
            logger.error(f"ROUTER: Error loading outcomes: {e}")
    
    def save_model(self):
        torch.save({
            "model_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_dim": 398,
            "hidden_dim": 256,
            "training_step_count": self.training_step_count,
            "exploration_rate": self.exploration_rate,
            "total_outcomes": len(self.outcomes)
        }, self.model_file)
        logger.info(f"ROUTER: Model saved to {self.model_file}")
        logger.info(f"  - Training steps: {self.training_step_count}")
        logger.info(f"  - Total outcomes: {len(self.outcomes)}")
        logger.info(f"  - Exploration rate: {self.exploration_rate:.3f}")
    
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
        
        logger.debug(f"ROUTER: Saved {len(data)} training examples to {training_file}")
    
    def _save_outcome(self, outcome: RoutingOutcome):
        """Append a single outcome to the JSONL file."""
        try:
            outcome_data = {
                "query": outcome.query,
                "query_type": outcome.query_type.value,
                "worker": outcome.decision.worker_specialty,
                "success": outcome.success,
                "confidence": outcome.decision.confidence,
                "accuracy_score": outcome.accuracy_score,
                "latency_ms": outcome.latency_ms,
                "timestamp": outcome.timestamp,
                "reasoning": outcome.decision.reasoning,
                "max_tree_depth": outcome.decision.max_tree_depth,
                "num_simulations": outcome.decision.num_simulations,
                "use_cache": outcome.decision.use_tree_cache
            }
            
            with open(self.outcomes_file, 'a') as f:
                f.write(json.dumps(outcome_data) + '\n')
            
            logger.debug(f"ROUTER: Saved outcome to {self.outcomes_file}")
        except Exception as e:
            logger.error(f"ROUTER: Failed to save outcome: {e}")
    
    def get_feedback_enhanced_stats(self) -> Dict[str, Any]:
        """Get router statistics enhanced with human feedback data."""
        
        feedback_stats = self.feedback_engine.get_statistics()
        
        worker_feedback_performance = {}
        for worker in self.idx_to_worker.values():
            worker_feedback_performance[worker] = self.feedback_engine.get_worker_performance(worker)
        
        return {
            "feedback_statistics": feedback_stats,
            "worker_feedback_performance": worker_feedback_performance,
            "worker_reward_adjustments": self.feedback_engine.worker_reward_adjustments,
            "step_quality_multiplier": self.feedback_engine.step_quality_multiplier,
            "total_feedback_count": feedback_stats.get("total_feedback", 0)
        }
