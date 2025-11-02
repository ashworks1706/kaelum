"""Cost tracking and metrics infrastructure."""

from typing import Dict, Any, Optional
from datetime import datetime
import json


class CostTracker:
    """Track and compare costs across commercial and local models."""
    
    def __init__(self):
        self.sessions = {}
        self.current_session = None
    
    def start_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start a new tracking session."""
        self.current_session = session_id
        self.sessions[session_id] = {
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
            "inferences": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0
        }
    
    def log_inference(
        self,
        model_type: str,
        tokens: int,
        latency_ms: float,
        cost: float,
        session_id: Optional[str] = None
    ):
        """Log a single inference."""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return
        
        session = self.sessions[sid]
        session["inferences"].append({
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "cost": cost
        })
        session["total_tokens"] += tokens
        session["total_cost"] += cost
        session["total_latency_ms"] += latency_ms
    
    def get_session_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a session."""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return {}
        
        session = self.sessions[sid]
        num_inferences = len(session["inferences"])
        
        return {
            "session_id": sid,
            "total_inferences": num_inferences,
            "total_tokens": session["total_tokens"],
            "total_cost": session["total_cost"],
            "total_latency_ms": session["total_latency_ms"],
            "avg_latency_ms": (
                session["total_latency_ms"] / num_inferences if num_inferences > 0 else 0
            ),
            "cost_per_1k_tokens": (
                (session["total_cost"] / session["total_tokens"]) * 1000
                if session["total_tokens"] > 0 else 0
            )
        }
    
    def calculate_savings(
        self,
        session_id: Optional[str] = None,
        commercial_rate_per_1m: float = 0.10  # Gemini 2.0 Flash blended rate
    ) -> Dict[str, Any]:
        """Calculate cost savings vs commercial LLM."""
        metrics = self.get_session_metrics(session_id)
        if not metrics:
            return {}
        
        tokens = metrics["total_tokens"]
        actual_cost = metrics["total_cost"]
        commercial_cost = (tokens / 1_000_000) * commercial_rate_per_1m
        savings = commercial_cost - actual_cost
        savings_pct = (savings / commercial_cost * 100) if commercial_cost > 0 else 0
        
        return {
            "actual_cost": actual_cost,
            "commercial_cost": commercial_cost,
            "savings": savings,
            "savings_percent": savings_pct,
            "tokens": tokens
        }
    
    def export_session(self, session_id: Optional[str] = None) -> str:
        """Export session data as JSON."""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return "{}"
        return json.dumps(self.sessions[sid], indent=2)
