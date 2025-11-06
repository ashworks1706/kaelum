"""Human feedback system for continuous improvement of all system components."""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from core.paths import DEFAULT_CACHE_DIR

logger = logging.getLogger("kaelum.feedback")

DEFAULT_FEEDBACK_DIR = str(Path(DEFAULT_CACHE_DIR) / "feedback")


@dataclass
class HumanFeedback:
    """Comprehensive human feedback on query execution."""
    
    # Query identification
    query: str
    query_hash: str
    timestamp: float
    
    # Overall feedback
    overall_liked: bool
    overall_rating: int  # 1-5 stars
    
    # Worker selection feedback
    worker_selected: str
    worker_correct: bool
    
    # Answer quality feedback
    answer_correct: bool
    answer_helpful: bool
    answer_complete: bool
    answer_rating: int
    
    # What was shown to user
    confidence_shown: float
    verification_passed: bool
    execution_time: float
    
    # Optional fields with defaults
    suggested_worker: Optional[str] = None
    steps_helpful: List[bool] = None
    steps_rating: List[int] = None  # 1-5 per step
    comment: Optional[str] = None
    
    def __post_init__(self):
        if self.steps_helpful is None:
            self.steps_helpful = []
        if self.steps_rating is None:
            self.steps_rating = []


@dataclass
class FeedbackStatistics:
    """Aggregated statistics from human feedback."""
    
    total_feedback: int = 0
    
    # Worker statistics
    worker_accuracy: Dict[str, float] = None
    worker_preference: Dict[str, int] = None
    worker_corrections: Dict[str, Dict[str, int]] = None  # {from_worker: {to_worker: count}}
    
    # Answer quality
    answer_accuracy: float = 0.0
    answer_helpfulness: float = 0.0
    answer_completeness: float = 0.0
    avg_answer_rating: float = 0.0
    
    # Step quality
    avg_step_rating: float = 0.0
    step_helpfulness_rate: float = 0.0
    
    # Overall
    overall_satisfaction: float = 0.0
    avg_overall_rating: float = 0.0
    
    def __post_init__(self):
        if self.worker_accuracy is None:
            self.worker_accuracy = {}
        if self.worker_preference is None:
            self.worker_preference = {}
        if self.worker_corrections is None:
            self.worker_corrections = {}


class HumanFeedbackEngine:
    """Engine for collecting and utilizing human feedback to improve the system."""
    
    def __init__(self, data_dir: str = DEFAULT_FEEDBACK_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.data_dir / "feedback.jsonl"
        self.stats_file = self.data_dir / "statistics.json"
        self.adjustments_file = self.data_dir / "reward_adjustments.json"
        
        self.feedback_history: List[HumanFeedback] = []
        self.statistics = FeedbackStatistics()
        
        # Reward adjustments based on feedback
        self.worker_reward_adjustments: Dict[str, float] = {
            "math": 0.0,
            "code": 0.0,
            "logic": 0.0,
            "factual": 0.0,
            "creative": 0.0,
            "analysis": 0.0
        }
        
        # Step quality adjustments (multiplier for intermediate rewards)
        self.step_quality_multiplier = 1.0
        
        self._load_data()
        
        logger.info(f"Human Feedback Engine initialized")
        logger.info(f"  - Feedback history: {len(self.feedback_history)} entries")
        logger.info(f"  - Data directory: {self.data_dir}")
    
    def submit_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Submit human feedback and update reward models."""
        
        logger.info("=" * 80)
        logger.info("HUMAN FEEDBACK: New feedback received")
        logger.info(f"  - Query: {feedback.query[:60]}...")
        logger.info(f"  - Overall liked: {feedback.overall_liked}")
        logger.info(f"  - Overall rating: {feedback.overall_rating}/5")
        logger.info(f"  - Worker selected: {feedback.worker_selected}")
        logger.info(f"  - Worker correct: {feedback.worker_correct}")
        logger.info(f"  - Answer correct: {feedback.answer_correct}")
        logger.info(f"  - Answer rating: {feedback.answer_rating}/5")
        
        # Add to history
        self.feedback_history.append(feedback)
        
        # Update statistics
        self._update_statistics(feedback)
        
        # Adjust reward models based on feedback
        adjustments = self._adjust_reward_models(feedback)
        
        # Save immediately
        self._save_feedback(feedback)
        self._save_statistics()
        self._save_adjustments()
        
        logger.info(f"HUMAN FEEDBACK: Processed successfully")
        logger.info(f"  - Total feedback count: {len(self.feedback_history)}")
        logger.info(f"  - Reward adjustments applied: {len(adjustments)}")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "feedback_id": feedback.query_hash,
            "adjustments_applied": adjustments,
            "statistics": asdict(self.statistics)
        }
    
    def _update_statistics(self, feedback: HumanFeedback):
        """Update aggregated statistics with new feedback."""
        
        # Initialize if needed
        if self.statistics.worker_accuracy is None:
            self.statistics.worker_accuracy = {}
        if self.statistics.worker_preference is None:
            self.statistics.worker_preference = {}
        if self.statistics.worker_corrections is None:
            self.statistics.worker_corrections = {}
        
        # Update total count
        self.statistics.total_feedback += 1
        n = self.statistics.total_feedback
        
        # Worker statistics
        worker = feedback.worker_selected
        if worker not in self.statistics.worker_accuracy:
            self.statistics.worker_accuracy[worker] = 0.0
            self.statistics.worker_preference[worker] = 0
            self.statistics.worker_corrections[worker] = {}
        
        # Update worker accuracy (running average)
        old_acc = self.statistics.worker_accuracy[worker]
        count = self.statistics.worker_preference[worker] + 1
        new_acc = (old_acc * self.statistics.worker_preference[worker] + float(feedback.worker_correct)) / count
        self.statistics.worker_accuracy[worker] = new_acc
        self.statistics.worker_preference[worker] += 1
        
        # Track corrections
        if not feedback.worker_correct and feedback.suggested_worker:
            suggested = feedback.suggested_worker
            if suggested not in self.statistics.worker_corrections[worker]:
                self.statistics.worker_corrections[worker][suggested] = 0
            self.statistics.worker_corrections[worker][suggested] += 1
        
        # Answer quality (running averages)
        self.statistics.answer_accuracy = (
            (self.statistics.answer_accuracy * (n - 1) + float(feedback.answer_correct)) / n
        )
        self.statistics.answer_helpfulness = (
            (self.statistics.answer_helpfulness * (n - 1) + float(feedback.answer_helpful)) / n
        )
        self.statistics.answer_completeness = (
            (self.statistics.answer_completeness * (n - 1) + float(feedback.answer_complete)) / n
        )
        self.statistics.avg_answer_rating = (
            (self.statistics.avg_answer_rating * (n - 1) + feedback.answer_rating) / n
        )
        
        # Step quality
        if feedback.steps_rating:
            avg_step = np.mean(feedback.steps_rating)
            self.statistics.avg_step_rating = (
                (self.statistics.avg_step_rating * (n - 1) + avg_step) / n
            )
        
        if feedback.steps_helpful:
            helpful_rate = np.mean([float(x) for x in feedback.steps_helpful])
            self.statistics.step_helpfulness_rate = (
                (self.statistics.step_helpfulness_rate * (n - 1) + helpful_rate) / n
            )
        
        # Overall satisfaction
        self.statistics.overall_satisfaction = (
            (self.statistics.overall_satisfaction * (n - 1) + float(feedback.overall_liked)) / n
        )
        self.statistics.avg_overall_rating = (
            (self.statistics.avg_overall_rating * (n - 1) + feedback.overall_rating) / n
        )
        
        logger.info(f"FEEDBACK STATS: Updated statistics")
        logger.info(f"  - {worker} accuracy: {new_acc:.3f}")
        logger.info(f"  - Answer accuracy: {self.statistics.answer_accuracy:.3f}")
        logger.info(f"  - Overall satisfaction: {self.statistics.overall_satisfaction:.3f}")
    
    def _adjust_reward_models(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Adjust reward models based on human feedback."""
        
        adjustments = {}
        
        # 1. WORKER SELECTION ADJUSTMENT
        # If human says worker was wrong, penalize that worker's rewards
        if not feedback.worker_correct:
            worker = feedback.worker_selected
            # Strong penalty for wrong worker selection
            adjustment = -0.03
            self.worker_reward_adjustments[worker] += adjustment
            adjustments[f"worker_{worker}_penalty"] = adjustment
            
            logger.info(f"REWARD ADJUSTMENT: Worker {worker} penalized by {adjustment:.3f}")
            
            # If human suggests correct worker, boost that worker
            if feedback.suggested_worker:
                suggested = feedback.suggested_worker
                boost = 0.05
                self.worker_reward_adjustments[suggested] += boost
                adjustments[f"worker_{suggested}_boost"] = boost
                logger.info(f"REWARD ADJUSTMENT: Worker {suggested} boosted by {boost:.3f}")
        
        # 2. ANSWER QUALITY ADJUSTMENT
        # Adjust final answer rewards based on correctness and quality
        if not feedback.answer_correct:
            # Wrong answer means final rewards should be lower
            worker = feedback.worker_selected
            adjustment = -0.05
            self.worker_reward_adjustments[worker] += adjustment
            adjustments[f"answer_quality_{worker}_penalty"] = adjustment
            
            logger.info(f"REWARD ADJUSTMENT: {worker} answer quality penalized by {adjustment:.3f}")
        
        elif feedback.answer_rating >= 4:
            # High-quality answer, boost final rewards
            worker = feedback.worker_selected
            boost = 0.02
            self.worker_reward_adjustments[worker] += boost
            adjustments[f"answer_quality_{worker}_boost"] = boost
            
            logger.info(f"REWARD ADJUSTMENT: {worker} answer quality boosted by {boost:.3f}")
        
        # 3. REASONING STEP ADJUSTMENT
        # Adjust intermediate step rewards based on per-step feedback
        if feedback.steps_rating:
            avg_step_rating = np.mean(feedback.steps_rating)
            
            # Steps rated highly -> boost partial rewards
            if avg_step_rating >= 4.0:
                old_mult = self.step_quality_multiplier
                self.step_quality_multiplier = min(1.2, self.step_quality_multiplier + 0.02)
                adjustments["step_quality_multiplier_boost"] = self.step_quality_multiplier - old_mult
                
                logger.info(f"REWARD ADJUSTMENT: Step quality multiplier boosted to {self.step_quality_multiplier:.3f}")
            
            # Steps rated poorly -> reduce partial rewards
            elif avg_step_rating <= 2.0:
                old_mult = self.step_quality_multiplier
                self.step_quality_multiplier = max(0.8, self.step_quality_multiplier - 0.02)
                adjustments["step_quality_multiplier_penalty"] = self.step_quality_multiplier - old_mult
                
                logger.info(f"REWARD ADJUSTMENT: Step quality multiplier reduced to {self.step_quality_multiplier:.3f}")
        
        # 4. CLAMP ADJUSTMENTS
        # Prevent adjustments from going too extreme
        for worker in self.worker_reward_adjustments:
            self.worker_reward_adjustments[worker] = np.clip(
                self.worker_reward_adjustments[worker],
                -0.15,  # Max penalty
                0.15    # Max boost
            )
        
        self.step_quality_multiplier = np.clip(self.step_quality_multiplier, 0.7, 1.3)
        
        return adjustments
    
    def get_adjusted_reward(self, worker_type: str, base_reward: float, 
                           is_partial: bool = False) -> float:
        """Get reward adjusted by human feedback."""
        
        # Apply worker-specific adjustment
        adjustment = self.worker_reward_adjustments.get(worker_type, 0.0)
        adjusted = base_reward + adjustment
        
        # Apply step quality multiplier for partial rewards
        if is_partial:
            adjusted *= self.step_quality_multiplier
        
        return max(0.0, min(1.0, adjusted))  # Clamp to [0, 1]
    
    def get_worker_performance(self, worker: str) -> Dict[str, Any]:
        """Get performance metrics for a specific worker."""
        
        if worker not in self.statistics.worker_accuracy:
            return {
                "accuracy": None,
                "total_feedback": 0,
                "reward_adjustment": self.worker_reward_adjustments.get(worker, 0.0),
                "corrections": {}
            }
        
        return {
            "accuracy": self.statistics.worker_accuracy[worker],
            "total_feedback": self.statistics.worker_preference[worker],
            "reward_adjustment": self.worker_reward_adjustments.get(worker, 0.0),
            "corrections": self.statistics.worker_corrections.get(worker, {})
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        
        return {
            "total_feedback": self.statistics.total_feedback,
            "worker_accuracy": self.statistics.worker_accuracy,
            "worker_preference": self.statistics.worker_preference,
            "worker_corrections": self.statistics.worker_corrections,
            "answer_accuracy": self.statistics.answer_accuracy,
            "answer_helpfulness": self.statistics.answer_helpfulness,
            "answer_completeness": self.statistics.answer_completeness,
            "avg_answer_rating": self.statistics.avg_answer_rating,
            "avg_step_rating": self.statistics.avg_step_rating,
            "step_helpfulness_rate": self.statistics.step_helpfulness_rate,
            "overall_satisfaction": self.statistics.overall_satisfaction,
            "avg_overall_rating": self.statistics.avg_overall_rating,
            "reward_adjustments": self.worker_reward_adjustments,
            "step_quality_multiplier": self.step_quality_multiplier
        }
    
    def _save_feedback(self, feedback: HumanFeedback):
        """Append feedback to JSONL file."""
        
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(asdict(feedback)) + "\n")
    
    def _save_statistics(self):
        """Save aggregated statistics."""
        
        with open(self.stats_file, "w") as f:
            json.dump(asdict(self.statistics), f, indent=2)
    
    def _save_adjustments(self):
        """Save reward adjustments."""
        
        data = {
            "worker_adjustments": self.worker_reward_adjustments,
            "step_quality_multiplier": self.step_quality_multiplier,
            "last_updated": time.time()
        }
        
        with open(self.adjustments_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load existing feedback data."""
        
        # Load feedback history
        if self.feedback_file.exists():
            with open(self.feedback_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        feedback = HumanFeedback(**data)
                        self.feedback_history.append(feedback)
        
        # Load statistics
        if self.stats_file.exists():
            with open(self.stats_file, "r") as f:
                data = json.load(f)
                self.statistics = FeedbackStatistics(**data)
        
        # Load adjustments
        if self.adjustments_file.exists():
            with open(self.adjustments_file, "r") as f:
                data = json.load(f)
                self.worker_reward_adjustments = data.get("worker_adjustments", self.worker_reward_adjustments)
                self.step_quality_multiplier = data.get("step_quality_multiplier", 1.0)
                
                logger.info(f"FEEDBACK: Loaded reward adjustments")
                for worker, adj in self.worker_reward_adjustments.items():
                    if abs(adj) > 0.001:
                        logger.info(f"  - {worker}: {adj:+.3f}")
                logger.info(f"  - Step quality multiplier: {self.step_quality_multiplier:.3f}")
