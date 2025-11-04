"""Neural Router (Kaelum Brain) - Adaptive policy network for routing decisions.

This implements the conceptual neural router described in the architecture docs:
- 1-2B parameter policy network (or lightweight MLP for local execution)
- Learns from historical routing outcomes
- Predicts optimal reasoning strategies dynamically
- Trades off speed vs. accuracy per query type
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import logging
    logger = logging.getLogger("kaelum.neural_router")
    logger.warning("PyTorch not installed. Neural router will use rule-based fallback.")
    logger.info("Install with: pip install torch")
    # Provide tiny stubs for `nn` and `F` so class definitions that reference
    # `nn.Module`, `nn.Linear`, etc. won't raise at import time when torch is
    # not installed. These stubs are inert and will raise if actually used.
    try:
        import types
    except Exception:
        types = None

    class _DummyModule:
        pass

    class _DummyLinear:
        def __init__(self, *a, **k):
            pass

    class _DummyLayerNorm:
        def __init__(self, *a, **k):
            pass

    def _dummy_relu(x):
        return x

    class _DummyDropout:
        def __init__(self, *a, **k):
            pass

    if types is not None:
        nn = types.SimpleNamespace(
            Linear=_DummyLinear,
            LayerNorm=_DummyLayerNorm,
            ReLU=_dummy_relu,
            Dropout=_DummyDropout,
            Sequential=lambda *args, **kwargs: None,
            Module=_DummyModule,
        )
        F = types.SimpleNamespace(
            softmax=lambda x, dim=-1: x,
            sigmoid=lambda x: x,
        )
    else:
        nn = None
        F = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Import base router components
from .router import (
    QueryType, ReasoningStrategy, RoutingDecision, RoutingOutcome, Router
)

logger = logging.getLogger("kaelum.neural_router")
logger.setLevel(logging.INFO)


@dataclass
class NeuralRoutingFeatures:
    """Features extracted from a query for neural routing."""
    # Query features
    query_embedding: np.ndarray  # 384-dim from sentence-transformers
    query_length: int
    query_complexity: float  # 0-1 score
    
    # Query type scores (from rule-based classifier)
    math_score: float
    logic_score: float
    code_score: float
    factual_score: float
    creative_score: float
    analysis_score: float
    
    # Context features
    has_numbers: bool
    has_operators: bool
    has_code_keywords: bool
    question_mark: bool
    
    # Historical performance (if available)
    similar_query_avg_accuracy: float = 0.5
    similar_query_avg_latency: float = 300.0
    
    def to_tensor(self) -> 'torch.Tensor':
        """Convert features to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Combine all features into a single vector
        categorical_features = np.array([
            self.query_length / 100.0,  # Normalize
            self.query_complexity,
            self.math_score,
            self.logic_score,
            self.code_score,
            self.factual_score,
            self.creative_score,
            self.analysis_score,
            float(self.has_numbers),
            float(self.has_operators),
            float(self.has_code_keywords),
            float(self.question_mark),
            self.similar_query_avg_accuracy,
            self.similar_query_avg_latency / 1000.0,  # Normalize to seconds
        ], dtype=np.float32)
        
        # Concatenate embedding + categorical
        full_features = np.concatenate([
            self.query_embedding,
            categorical_features
        ])
        
        return torch.from_numpy(full_features).float()
    
    @property
    def feature_dim(self) -> int:
        """Total feature dimension."""
        return len(self.query_embedding) + 14  # 384 + 14 categorical


class PolicyNetwork(nn.Module):
    """Lightweight neural policy network for routing decisions.
    
    Architecture:
    - Input: Query features (embeddings + metadata)
    - Hidden layers: MLP with residual connections
    - Outputs: 
        - Strategy probabilities (5 classes)
        - Max reflection iterations (regression, 0-3)
        - Use symbolic verification (binary)
        - Use factual verification (binary)
        - Confidence threshold (regression, 0.5-0.95)
    """
    
    def __init__(self, input_dim: int = 398, hidden_dim: int = 256, num_strategies: int = 5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        
        # Feature encoder (3-layer MLP with residual connections)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Output heads
        self.strategy_head = nn.Linear(hidden_dim, num_strategies)  # Softmax over strategies
        self.reflection_head = nn.Linear(hidden_dim, 1)  # Regression for max iterations
        self.symbolic_head = nn.Linear(hidden_dim, 1)  # Binary for symbolic verification
        self.factual_head = nn.Linear(hidden_dim, 1)  # Binary for factual verification
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Regression for threshold
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy network.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary of outputs for each head
        """
        # Encode features
        h = self.encoder(x)
        
        # Hidden layers with residual connections
        h1 = self.hidden1(h)
        h = h + h1  # Residual
        
        h2 = self.hidden2(h)
        h = h + h2  # Residual
        
        # Multiple output heads
        outputs = {
            'strategy_logits': self.strategy_head(h),  # [batch, num_strategies]
            'reflection_logits': self.reflection_head(h),  # [batch, 1]
            'symbolic_logits': self.symbolic_head(h),  # [batch, 1]
            'factual_logits': self.factual_head(h),  # [batch, 1]
            'confidence_logits': self.confidence_head(h),  # [batch, 1]
        }
        
        return outputs
    
    def predict_routing(self, x: torch.Tensor) -> Dict[str, Any]:
        """Predict routing decision from features.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predicted routing parameters
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Strategy: argmax over softmax
            strategy_probs = F.softmax(outputs['strategy_logits'], dim=-1)
            strategy_idx = torch.argmax(strategy_probs, dim=-1).item()
            strategy_conf = strategy_probs[0, strategy_idx].item()
            
            # Reflection iterations: round and clip to [0, 3]
            reflection = torch.clamp(
                torch.round(torch.sigmoid(outputs['reflection_logits']) * 3),
                0, 3
            ).int().item()
            
            # Binary decisions: sigmoid > 0.5
            use_symbolic = (torch.sigmoid(outputs['symbolic_logits']) > 0.5).item()
            use_factual = (torch.sigmoid(outputs['factual_logits']) > 0.5).item()
            
            # Confidence threshold: scale sigmoid to [0.5, 0.95]
            confidence_thresh = (
                0.5 + 0.45 * torch.sigmoid(outputs['confidence_logits'])
            ).item()
            
            return {
                'strategy_idx': strategy_idx,
                'strategy_confidence': strategy_conf,
                'max_reflection_iterations': reflection,
                'use_symbolic_verification': use_symbolic,
                'use_factual_verification': use_factual,
                'confidence_threshold': confidence_thresh,
            }
class NeuralRouter:
    """Neural router using learned policy network.
    
    This is the "Kaelum Brain" - a learned controller that adapts routing
    decisions based on query characteristics and historical performance.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        data_dir: str = ".kaelum/neural_routing",
        fallback_to_rules: bool = True,
        device: str = "cpu"
    ):
        """Initialize neural router.
        
        Args:
            model_path: Path to saved model checkpoint
            data_dir: Directory for training data and checkpoints
            fallback_to_rules: Use rule-based router if neural model unavailable
            device: Device for inference ('cpu' or 'cuda')
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.fallback_to_rules = fallback_to_rules
        self.device = device if TORCH_AVAILABLE else "cpu"
        
        # Strategy mapping
        self.strategy_idx_to_enum = [
            ReasoningStrategy.SYMBOLIC_HEAVY,
            ReasoningStrategy.FACTUAL_HEAVY,
            ReasoningStrategy.BALANCED,
            ReasoningStrategy.FAST,
            ReasoningStrategy.DEEP,
        ]
        
        # Initialize components
        self.encoder = None
        self._embedding_attempted = False
        # Don't load embeddings at init - do it lazily on first use
        logger.info("Neural router initialized (embeddings will load on first use)")
        
        self.policy_network = None
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            # Initialize policy network
            self.policy_network = PolicyNetwork(
                input_dim=398,  # 384 (embedding) + 14 (categorical)
                hidden_dim=256,
                num_strategies=5
            )
            
            # Load model if path provided
            if model_path and Path(model_path).exists():
                self._load_model(model_path)
            else:
                # Try loading from default location
                default_path = self.data_dir / "neural_router.pt"
                if default_path.exists():
                    self._load_model(str(default_path))
        
        # Fallback rule-based router
        self.rule_router = None
        if fallback_to_rules:
            self.rule_router = Router(learning_enabled=True, data_dir=str(self.data_dir.parent / "routing"))
        
        logger.info("=" * 60)
        logger.info("Neural Router (Kaelum Brain) initialized")
        logger.info(f"PyTorch available: {TORCH_AVAILABLE}")
        logger.info(f"Model loaded: {self.model_loaded}")
        logger.info(f"Embeddings available: {EMBEDDINGS_AVAILABLE}")
        logger.info(f"Fallback to rules: {fallback_to_rules}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)
    
    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Route a query using neural policy network.
        
        Args:
            query: Input query to route
            context: Optional context
            
        Returns:
            RoutingDecision with predicted strategy
        """
        start_time = time.time()
        
        logger.info("-" * 60)
        logger.info(f"NEURAL ROUTING: {query[:100]}...")
        
        # Use neural model if available
        if self.model_loaded and TORCH_AVAILABLE and self.encoder:
            try:
                decision = self._neural_route(query, context)
                routing_time = (time.time() - start_time) * 1000
                logger.info(f"  Neural routing time: {routing_time:.2f}ms")
                logger.info("-" * 60)
                return decision
            except Exception as e:
                logger.warning(f"Neural routing failed: {e}")
                logger.info("Falling back to rule-based router")
        
        # Fallback to rule-based
        if self.rule_router:
            logger.info("Using rule-based router (neural model not available)")
            return self.rule_router.route(query, context)
        else:
            # Ultimate fallback: balanced strategy
            logger.warning("No routing available - using default balanced strategy")
            return RoutingDecision(
                query_type=QueryType.UNKNOWN,
                strategy=ReasoningStrategy.BALANCED,
                max_reflection_iterations=2,
                use_symbolic_verification=True,
                use_factual_verification=True,
                confidence_threshold=0.75,
                reasoning="Default strategy (no routing available)"
            )
    
    def _neural_route(self, query: str, context: Optional[Dict]) -> RoutingDecision:
        """Perform neural routing using policy network.
        
        Args:
            query: Input query
            context: Optional context
            
        Returns:
            RoutingDecision from neural model
        """
        # Step 1: Extract features
        features = self._extract_features(query, context)
        
        # Step 2: Convert to tensor
        x = features.to_tensor().unsqueeze(0)  # Add batch dimension
        
        if self.device == "cuda":
            x = x.cuda()
        
        # Step 3: Get prediction from neural network
        prediction = self.policy_network.predict_routing(x)
        
        # Step 4: Convert to routing decision
        strategy = self.strategy_idx_to_enum[prediction['strategy_idx']]
        
        # Infer query type from scores
        type_scores = {
            QueryType.MATH: features.math_score,
            QueryType.LOGIC: features.logic_score,
            QueryType.CODE: features.code_score,
            QueryType.FACTUAL: features.factual_score,
            QueryType.CREATIVE: features.creative_score,
            QueryType.ANALYSIS: features.analysis_score,
        }
        query_type = max(type_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"  Query Type: {query_type.value} (neural inference)")
        logger.info(f"  Strategy: {strategy.value} (confidence: {prediction['strategy_confidence']:.2f})")
        logger.info(f"  Config: reflection={prediction['max_reflection_iterations']}, "
                   f"symbolic={prediction['use_symbolic_verification']}, "
                   f"factual={prediction['use_factual_verification']}")
        
        return RoutingDecision(
            query_type=query_type,
            strategy=strategy,
            max_reflection_iterations=prediction['max_reflection_iterations'],
            use_symbolic_verification=prediction['use_symbolic_verification'],
            use_factual_verification=prediction['use_factual_verification'],
            confidence_threshold=prediction['confidence_threshold'],
            reasoning=f"Neural router prediction (confidence: {prediction['strategy_confidence']:.2f})",
            complexity_score=features.query_complexity
        )
    
    def _extract_features(self, query: str, context: Optional[Dict]) -> NeuralRoutingFeatures:
        """Extract features from query for neural routing.
        
        Args:
            query: Input query
            context: Optional context
            
        Returns:
            NeuralRoutingFeatures object
        """
        # Get query embedding (lazy load)
        if not self._embedding_attempted and EMBEDDINGS_AVAILABLE:
            try:
                logger.info("Loading sentence embedding model (first use)...")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Embedding model loaded")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Using zero embeddings (degraded mode)")
            self._embedding_attempted = True
        
        if self.encoder:
            query_embedding = self.encoder.encode(query, show_progress_bar=False)
        else:
            # Fallback: zero embedding
            query_embedding = np.zeros(384, dtype=np.float32)
        
        # Calculate query type scores using rule-based classifier
        query_lower = query.lower()
        
        # Math score
        math_keywords = ["calculate", "solve", "equation", "sum", "multiply", "divide"]
        math_score = sum(0.2 for kw in math_keywords if kw in query_lower)
        math_score += 0.3 if any(c.isdigit() for c in query) else 0
        math_score = min(math_score, 1.0)
        
        # Logic score
        logic_keywords = ["if", "then", "therefore", "prove", "valid"]
        logic_score = sum(0.2 for kw in logic_keywords if kw in query_lower)
        logic_score = min(logic_score, 1.0)
        
        # Code score
        code_keywords = ["function", "code", "implement", "algorithm", "python"]
        code_score = sum(0.2 for kw in code_keywords if kw in query_lower)
        code_score = min(code_score, 1.0)
        
        # Factual score
        factual_keywords = ["who", "when", "where", "what", "history", "define"]
        factual_score = sum(0.15 for kw in factual_keywords if kw in query_lower)
        factual_score = min(factual_score, 1.0)
        
        # Creative score
        creative_keywords = ["poem", "story", "imagine", "create", "design"]
        creative_score = sum(0.2 for kw in creative_keywords if kw in query_lower)
        creative_score = min(creative_score, 1.0)
        
        # Analysis score
        analysis_keywords = ["analyze", "compare", "evaluate", "assess"]
        analysis_score = sum(0.25 for kw in analysis_keywords if kw in query_lower)
        analysis_score = min(analysis_score, 1.0)
        
        # Calculate complexity
        words = query.split()
        complexity = min(len(words) / 50.0, 0.3)  # Length factor
        complexity += min(query.count('(') + query.count('[') * 0.03, 0.2)  # Nesting
        complexity = min(complexity, 1.0)
        
        # Context features
        has_numbers = any(c.isdigit() for c in query)
        has_operators = any(op in query for op in ['+', '-', '*', '/', '='])
        has_code_keywords = any(kw in query_lower for kw in ['def', 'class', 'return', 'function'])
        question_mark = '?' in query
        
        return NeuralRoutingFeatures(
            query_embedding=query_embedding,
            query_length=len(query),
            query_complexity=complexity,
            math_score=math_score,
            logic_score=logic_score,
            code_score=code_score,
            factual_score=factual_score,
            creative_score=creative_score,
            analysis_score=analysis_score,
            has_numbers=has_numbers,
            has_operators=has_operators,
            has_code_keywords=has_code_keywords,
            question_mark=question_mark,
        )
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            self.policy_network.eval()
            self.model_loaded = True
            logger.info(f"✓ Neural router model loaded from {model_path}")
            
            if 'training_info' in checkpoint:
                info = checkpoint['training_info']
                logger.info(f"  Model trained on {info.get('num_samples', 'unknown')} samples")
                logger.info(f"  Training accuracy: {info.get('accuracy', 0):.2f}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def save_model(self, model_path: Optional[str] = None, training_info: Optional[Dict] = None):
        """Save trained model to checkpoint.
        
        Args:
            model_path: Path to save model (default: data_dir/neural_router.pt)
            training_info: Optional training metadata
        """
        if not TORCH_AVAILABLE or self.policy_network is None:
            logger.warning("Cannot save model: PyTorch or policy network not available")
            return
        
        if model_path is None:
            model_path = str(self.data_dir / "neural_router.pt")
        
        checkpoint = {
            'model_state_dict': self.policy_network.state_dict(),
            'model_config': {
                'input_dim': self.policy_network.input_dim,
                'hidden_dim': self.policy_network.hidden_dim,
                'num_strategies': self.policy_network.num_strategies,
            },
            'training_info': training_info or {},
            'timestamp': time.time(),
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"✓ Model saved to {model_path}")
    
    def record_outcome(self, decision: RoutingDecision, result: Dict[str, Any]):
        """Record routing outcome for future training.
        
        Args:
            decision: Routing decision made
            result: Result from orchestrator
        """
        # Delegate to rule-based router's outcome recording
        if self.rule_router:
            self.rule_router.record_outcome(decision, result)
