"""Training pipeline for the Neural Router (Kaelum Brain).

Trains the policy network on historical routing outcomes to learn optimal
strategies for different query types.
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Cannot train neural router.")
    print("   Install with: pip install torch")

from .neural_router import NeuralRouter, NeuralRoutingFeatures, PolicyNetwork
from .router import QueryType, ReasoningStrategy, RoutingOutcome

logger = logging.getLogger("kaelum.neural_router_trainer")
logger.setLevel(logging.INFO)


@dataclass
class TrainingSample:
    """Single training sample for neural router."""
    features: NeuralRoutingFeatures
    target_strategy: int  # Index of strategy
    target_reflection: int  # 0-3
    target_symbolic: bool
    target_factual: bool
    target_confidence: float
    accuracy_score: float  # For weighting samples
    

class RoutingDataset(Dataset):
    """PyTorch dataset for routing training data."""
    
    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        
        # Input features
        x = sample.features.to_tensor()
        
        # Target labels
        targets = {
            'strategy': torch.tensor(sample.target_strategy, dtype=torch.long),
            'reflection': torch.tensor(sample.target_reflection, dtype=torch.float),
            'symbolic': torch.tensor(float(sample.target_symbolic), dtype=torch.float),
            'factual': torch.tensor(float(sample.target_factual), dtype=torch.float),
            'confidence': torch.tensor(sample.target_confidence, dtype=torch.float),
            'weight': torch.tensor(sample.accuracy_score, dtype=torch.float),  # Sample weight
        }
        
        return x, targets


class NeuralRouterTrainer:
    """Trainer for neural router policy network."""
    
    def __init__(
        self,
        neural_router: NeuralRouter,
        outcomes_file: Optional[str] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """Initialize trainer.
        
        Args:
            neural_router: NeuralRouter instance to train
            outcomes_file: Path to outcomes JSONL file
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device for training ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training. Install with: pip install torch")
        
        self.neural_router = neural_router
        self.device = device
        self.batch_size = batch_size
        
        # Get outcomes file path
        if outcomes_file:
            self.outcomes_file = Path(outcomes_file)
        else:
            # Default to router's data directory
            router_data_dir = neural_router.data_dir.parent / "routing"
            self.outcomes_file = router_data_dir / "outcomes.jsonl"
        
        # Optimizer
        if neural_router.policy_network:
            self.optimizer = optim.AdamW(
                neural_router.policy_network.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            raise RuntimeError("Neural router policy network not initialized")
        
        # Loss functions
        self.strategy_loss_fn = nn.CrossEntropyLoss(reduction='none')  # For weighting
        self.reflection_loss_fn = nn.MSELoss(reduction='none')
        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.confidence_loss_fn = nn.MSELoss(reduction='none')
        
        logger.info("Neural Router Trainer initialized")
        logger.info(f"Outcomes file: {self.outcomes_file}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
    
    def load_training_data(self, min_samples: int = 50) -> List[TrainingSample]:
        """Load and prepare training data from outcomes file.
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            List of TrainingSample objects
        """
        if not self.outcomes_file.exists():
            raise FileNotFoundError(f"Outcomes file not found: {self.outcomes_file}")
        
        logger.info(f"Loading training data from {self.outcomes_file}")
        
        samples = []
        strategy_to_idx = {
            ReasoningStrategy.SYMBOLIC_HEAVY: 0,
            ReasoningStrategy.FACTUAL_HEAVY: 1,
            ReasoningStrategy.BALANCED: 2,
            ReasoningStrategy.FAST: 3,
            ReasoningStrategy.DEEP: 4,
        }
        
        with open(self.outcomes_file, 'r') as f:
            for line in f:
                try:
                    outcome_data = json.loads(line)
                    
                    # Extract query and results
                    query = outcome_data.get('query', '')
                    if not query:
                        continue
                    
                    # Parse strategy
                    strategy_str = outcome_data.get('strategy', 'balanced')
                    try:
                        strategy = ReasoningStrategy[strategy_str.upper()]
                    except (KeyError, AttributeError):
                        strategy = ReasoningStrategy.BALANCED
                    
                    strategy_idx = strategy_to_idx.get(strategy, 2)  # Default to BALANCED
                    
                    # Extract decision parameters (if available)
                    decision = outcome_data.get('decision', {})
                    if isinstance(decision, dict):
                        reflection = decision.get('max_reflection_iterations', 2)
                        symbolic = decision.get('use_symbolic_verification', True)
                        factual = decision.get('use_factual_verification', True)
                        confidence = decision.get('confidence_threshold', 0.75)
                    else:
                        # Use defaults based on strategy
                        reflection = 2 if strategy == ReasoningStrategy.DEEP else 1
                        symbolic = strategy in [ReasoningStrategy.SYMBOLIC_HEAVY, ReasoningStrategy.BALANCED, ReasoningStrategy.DEEP]
                        factual = strategy in [ReasoningStrategy.FACTUAL_HEAVY, ReasoningStrategy.BALANCED, ReasoningStrategy.DEEP]
                        confidence = 0.75
                    
                    # Performance metrics
                    accuracy = outcome_data.get('accuracy_score', 0.5)
                    
                    # Extract features
                    features = self.neural_router._extract_features(query, None)
                    
                    # Create training sample
                    sample = TrainingSample(
                        features=features,
                        target_strategy=strategy_idx,
                        target_reflection=min(reflection, 3),
                        target_symbolic=symbolic,
                        target_factual=factual,
                        target_confidence=confidence,
                        accuracy_score=accuracy,
                    )
                    
                    samples.append(sample)
                    
                except Exception as e:
                    logger.debug(f"Skipping invalid outcome: {e}")
                    continue
        
        logger.info(f"Loaded {len(samples)} training samples")
        
        if len(samples) < min_samples:
            logger.warning(f"Only {len(samples)} samples available (minimum: {min_samples})")
            logger.warning("Consider generating more synthetic data or collecting more outcomes")
        
        return samples
    
    def train(
        self,
        num_epochs: int = 50,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """Train the neural router policy network.
        
        Args:
            num_epochs: Number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for N epochs
            save_best: Save best model during training
            
        Returns:
            Training history and metrics
        """
        # Load training data
        samples = self.load_training_data()
        
        if len(samples) == 0:
            raise ValueError("No training samples available")
        
        # Split into train/validation
        np.random.shuffle(samples)
        split_idx = int(len(samples) * (1 - validation_split))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")
        
        # Create data loaders
        train_dataset = RoutingDataset(train_samples)
        val_dataset = RoutingDataset(val_samples)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }
        
        logger.info("=" * 60)
        logger.info("Starting neural router training")
        logger.info("=" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_accuracy'].append(train_metrics['strategy_accuracy'])
            history['val_accuracy'].append(val_metrics['strategy_accuracy'])
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Strategy Acc: {train_metrics['strategy_accuracy']:.2%}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Strategy Acc: {val_metrics['strategy_accuracy']:.2%}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                if save_best:
                    self.neural_router.save_model(
                        training_info={
                            'num_samples': len(samples),
                            'epoch': epoch + 1,
                            'accuracy': val_metrics['strategy_accuracy'],
                            'val_loss': val_metrics['loss'],
                        }
                    )
                    logger.info("  ✓ Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Learning rate scheduling
            self.scheduler.step()
        
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 60)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.neural_router.policy_network.train()
        
        total_loss = 0.0
        strategy_correct = 0
        total_samples = 0
        
        for batch_x, batch_targets in train_loader:
            # Move to device
            batch_x = batch_x.to(self.device)
            for key in batch_targets:
                batch_targets[key] = batch_targets[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.neural_router.policy_network(batch_x)
            
            # Calculate losses with sample weighting
            weights = batch_targets['weight']
            
            # Strategy loss (classification)
            strategy_loss = self.strategy_loss_fn(
                outputs['strategy_logits'],
                batch_targets['strategy']
            )
            strategy_loss = (strategy_loss * weights).mean()
            
            # Reflection loss (regression)
            reflection_pred = torch.sigmoid(outputs['reflection_logits'].squeeze()) * 3
            reflection_loss = self.reflection_loss_fn(
                reflection_pred,
                batch_targets['reflection']
            )
            reflection_loss = (reflection_loss * weights).mean()
            
            # Binary verification losses
            symbolic_loss = self.binary_loss_fn(
                outputs['symbolic_logits'].squeeze(),
                batch_targets['symbolic']
            )
            symbolic_loss = (symbolic_loss * weights).mean()
            
            factual_loss = self.binary_loss_fn(
                outputs['factual_logits'].squeeze(),
                batch_targets['factual']
            )
            factual_loss = (factual_loss * weights).mean()
            
            # Confidence loss (regression, scaled to [0.5, 0.95])
            confidence_pred = 0.5 + 0.45 * torch.sigmoid(outputs['confidence_logits'].squeeze())
            confidence_loss = self.confidence_loss_fn(
                confidence_pred,
                batch_targets['confidence']
            )
            confidence_loss = (confidence_loss * weights).mean()
            
            # Combined loss
            loss = (
                strategy_loss +
                0.3 * reflection_loss +
                0.2 * symbolic_loss +
                0.2 * factual_loss +
                0.3 * confidence_loss
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.neural_router.policy_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Strategy accuracy
            strategy_pred = torch.argmax(outputs['strategy_logits'], dim=-1)
            strategy_correct += (strategy_pred == batch_targets['strategy']).sum().item()
            total_samples += batch_x.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'strategy_accuracy': strategy_correct / total_samples if total_samples > 0 else 0,
        }
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.neural_router.policy_network.eval()
        
        total_loss = 0.0
        strategy_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_targets in val_loader:
                # Move to device
                batch_x = batch_x.to(self.device)
                for key in batch_targets:
                    batch_targets[key] = batch_targets[key].to(self.device)
                
                # Forward pass
                outputs = self.neural_router.policy_network(batch_x)
                
                # Calculate losses (same as training, but no weighting for validation)
                strategy_loss = self.strategy_loss_fn(
                    outputs['strategy_logits'],
                    batch_targets['strategy']
                ).mean()
                
                reflection_pred = torch.sigmoid(outputs['reflection_logits'].squeeze()) * 3
                reflection_loss = self.reflection_loss_fn(
                    reflection_pred,
                    batch_targets['reflection']
                ).mean()
                
                symbolic_loss = self.binary_loss_fn(
                    outputs['symbolic_logits'].squeeze(),
                    batch_targets['symbolic']
                ).mean()
                
                factual_loss = self.binary_loss_fn(
                    outputs['factual_logits'].squeeze(),
                    batch_targets['factual']
                ).mean()
                
                confidence_pred = 0.5 + 0.45 * torch.sigmoid(outputs['confidence_logits'].squeeze())
                confidence_loss = self.confidence_loss_fn(
                    confidence_pred,
                    batch_targets['confidence']
                ).mean()
                
                loss = (
                    strategy_loss +
                    0.3 * reflection_loss +
                    0.2 * symbolic_loss +
                    0.2 * factual_loss +
                    0.3 * confidence_loss
                )
                
                # Track metrics
                total_loss += loss.item()
                
                # Strategy accuracy
                strategy_pred = torch.argmax(outputs['strategy_logits'], dim=-1)
                strategy_correct += (strategy_pred == batch_targets['strategy']).sum().item()
                total_samples += batch_x.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'strategy_accuracy': strategy_correct / total_samples if total_samples > 0 else 0,
        }
    
    def generate_synthetic_data(
        self,
        num_samples: int = 500,
        save_to_outcomes: bool = True
    ) -> List[TrainingSample]:
        """Generate synthetic training data for bootstrapping.
        
        Args:
            num_samples: Number of synthetic samples to generate
            save_to_outcomes: Save to outcomes file for future use
            
        Returns:
            List of synthetic training samples
        """
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        # Define query templates for each type
        templates = {
            QueryType.MATH: [
                "Calculate {a} + {b} * {c}",
                "Solve for x: {a}x + {b} = {c}",
                "What is {a}% of {b}?",
                "Find the area of a circle with radius {a}",
                "If I have {a} items at ${b} each, what's the total?",
            ],
            QueryType.LOGIC: [
                "If all A are B and all B are C, are all A also C?",
                "Given that P implies Q, and Q implies R, what can we conclude?",
                "Is the following argument valid: {premise1}, {premise2}, therefore {conclusion}?",
            ],
            QueryType.CODE: [
                "Write a function to reverse a string",
                "Implement a binary search algorithm",
                "How do I sort a list in Python?",
                "Debug this code: {code_snippet}",
            ],
            QueryType.FACTUAL: [
                "Who was the first president of the United States?",
                "When was the Eiffel Tower built?",
                "What is the capital of {country}?",
                "Who invented the telephone?",
            ],
            QueryType.CREATIVE: [
                "Write a haiku about {topic}",
                "Tell me a story about {character}",
                "Imagine a world where {scenario}",
            ],
            QueryType.ANALYSIS: [
                "Compare {topic1} and {topic2}",
                "Analyze the pros and cons of {topic}",
                "What are the effects of {topic}?",
            ],
        }
        
        strategy_to_idx = {
            ReasoningStrategy.SYMBOLIC_HEAVY: 0,
            ReasoningStrategy.FACTUAL_HEAVY: 1,
            ReasoningStrategy.BALANCED: 2,
            ReasoningStrategy.FAST: 3,
            ReasoningStrategy.DEEP: 4,
        }
        
        # Optimal strategies for each query type
        optimal_strategies = {
            QueryType.MATH: ReasoningStrategy.SYMBOLIC_HEAVY,
            QueryType.LOGIC: ReasoningStrategy.DEEP,
            QueryType.CODE: ReasoningStrategy.DEEP,
            QueryType.FACTUAL: ReasoningStrategy.FACTUAL_HEAVY,
            QueryType.CREATIVE: ReasoningStrategy.FAST,
            QueryType.ANALYSIS: ReasoningStrategy.BALANCED,
        }
        
        samples = []
        outcomes_data = []
        
        for _ in range(num_samples):
            # Random query type
            query_type = np.random.choice(list(templates.keys()))
            template = np.random.choice(templates[query_type])
            
            # Fill template with random values
            query = template.format(
                a=np.random.randint(1, 100),
                b=np.random.randint(1, 100),
                c=np.random.randint(1, 100),
                topic=np.random.choice(["nature", "technology", "love", "adventure"]),
                country=np.random.choice(["France", "Japan", "Brazil", "Kenya"]),
                character=np.random.choice(["robot", "wizard", "explorer", "artist"]),
                topic1="option A",
                topic2="option B",
                premise1="P1",
                premise2="P2",
                conclusion="C",
                code_snippet="code",
                scenario="scenario"
            )
            
            # Get optimal strategy
            strategy = optimal_strategies[query_type]
            strategy_idx = strategy_to_idx[strategy]
            
            # Strategy-specific parameters
            if strategy == ReasoningStrategy.SYMBOLIC_HEAVY:
                reflection = 2
                symbolic = True
                factual = False
                confidence = 0.85
            elif strategy == ReasoningStrategy.FACTUAL_HEAVY:
                reflection = 1
                symbolic = False
                factual = True
                confidence = 0.80
            elif strategy == ReasoningStrategy.BALANCED:
                reflection = 2
                symbolic = True
                factual = True
                confidence = 0.75
            elif strategy == ReasoningStrategy.FAST:
                reflection = 0
                symbolic = True
                factual = False
                confidence = 0.70
            else:  # DEEP
                reflection = 3
                symbolic = True
                factual = True
                confidence = 0.90
            
            # Simulate accuracy based on strategy fit
            accuracy = 0.85 + np.random.normal(0, 0.05)
            accuracy = np.clip(accuracy, 0.5, 1.0)
            
            # Extract features
            features = self.neural_router._extract_features(query, None)
            
            # Create sample
            sample = TrainingSample(
                features=features,
                target_strategy=strategy_idx,
                target_reflection=reflection,
                target_symbolic=symbolic,
                target_factual=factual,
                target_confidence=confidence,
                accuracy_score=accuracy,
            )
            samples.append(sample)
            
            # Create outcome data for saving
            if save_to_outcomes:
                outcome = {
                    'query': query,
                    'query_type': query_type.value,
                    'strategy': strategy.value,
                    'decision': {
                        'max_reflection_iterations': reflection,
                        'use_symbolic_verification': symbolic,
                        'use_factual_verification': factual,
                        'confidence_threshold': confidence,
                    },
                    'success': accuracy > 0.75,
                    'accuracy_score': float(accuracy),
                    'latency_ms': 300.0,
                    'cost': 0.00001,
                    'symbolic_passed': symbolic and accuracy > 0.8,
                    'factual_passed': factual and accuracy > 0.8,
                    'reflection_iterations': reflection > 0,
                    'timestamp': time.time(),
                }
                outcomes_data.append(outcome)
        
        # Save to outcomes file
        if save_to_outcomes and outcomes_data:
            self.outcomes_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.outcomes_file, 'a') as f:
                for outcome in outcomes_data:
                    f.write(json.dumps(outcome) + '\n')
            logger.info(f"✓ Saved {len(outcomes_data)} synthetic outcomes to {self.outcomes_file}")
        
        logger.info(f"✓ Generated {len(samples)} synthetic training samples")
        return samples
