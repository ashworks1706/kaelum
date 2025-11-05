from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import time


class TokenCounter:
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count using character-based approximation.
        Average: 1 token ≈ 4 characters for English text.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    @staticmethod
    def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """Count tokens across multiple messages."""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            total += TokenCounter.count_tokens(content)
            total += 4
        return total


class AnalyticsDashboard:
    
    def __init__(self, storage_dir: str = ".kaelum/analytics"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.storage_dir / "metrics.jsonl"
        self.summary_file = self.storage_dir / "summary.json"
    
    def record_query(self, query_data: Dict[str, Any]):
        """Record detailed query execution metrics."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(query_data) + '\n')
        self._update_summary(query_data)
    
    def _update_summary(self, query_data: Dict[str, Any]):
        """Maintain running summary statistics."""
        summary = self._load_summary()
        
        summary['total_queries'] = summary.get('total_queries', 0) + 1
        summary['total_tokens'] = summary.get('total_tokens', 0) + query_data.get('tokens', 0)
        summary['total_time_ms'] = summary.get('total_time_ms', 0) + query_data.get('time_ms', 0)
        summary['total_simulations'] = summary.get('total_simulations', 0) + query_data.get('num_simulations', 0)
        
        worker = query_data.get('worker', 'unknown')
        summary['by_worker'] = summary.get('by_worker', {})
        summary['by_worker'][worker] = summary['by_worker'].get(worker, 0) + 1
        
        if query_data.get('verification_passed'):
            summary['verified_queries'] = summary.get('verified_queries', 0) + 1
        
        if query_data.get('cache_hit'):
            summary['cache_hits'] = summary.get('cache_hits', 0) + 1
        
        summary['last_updated'] = datetime.now().isoformat()
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _load_summary(self) -> Dict[str, Any]:
        """Load existing summary or create new one."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # If file is corrupted, start fresh
                return {}
        return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        summary = self._load_summary()
        
        if summary.get('total_queries', 0) > 0:
            summary['avg_time_ms'] = summary['total_time_ms'] / summary['total_queries']
            summary['avg_tokens'] = summary['total_tokens'] / summary['total_queries']
            summary['verification_rate'] = summary.get('verified_queries', 0) / summary['total_queries']
            summary['cache_hit_rate'] = summary.get('cache_hits', 0) / summary['total_queries']
        
        return summary
    
    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query records."""
        if not self.metrics_file.exists():
            return []
        
        queries = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        
        return queries[-limit:]


class CostTracker:
    def __init__(self):
        self.sessions = {}
        self.current_session = None
        self.token_counter = TokenCounter()
        self.analytics = AnalyticsDashboard()
    
    def start_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.current_session = session_id
        self.sessions[session_id] = {
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
            "inferences": [],
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "reflection_iterations": 0
        }
    
    def log_inference(
        self,
        input_text: str,
        output_text: str,
        latency_ms: float,
        worker_type: str = "unknown",
        verification_passed: bool = False,
        cache_hit: bool = False,
        num_simulations: int = 0,
        session_id: Optional[str] = None
    ):
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            self.start_session(sid or "default")
            sid = self.current_session
        
        session = self.sessions[sid]
        
        input_tokens = self.token_counter.count_tokens(input_text)
        output_tokens = self.token_counter.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        cost = 0.0
        
        inference_record = {
            "timestamp": datetime.now().isoformat(),
            "worker_type": worker_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "cost": cost,
            "verification_passed": verification_passed,
            "cache_hit": cache_hit,
            "num_simulations": num_simulations
        }
        
        session["inferences"].append(inference_record)
        session["total_tokens"] += total_tokens
        session["total_input_tokens"] += input_tokens
        session["total_output_tokens"] += output_tokens
        session["total_cost"] += cost
        session["total_latency_ms"] += latency_ms
        
        if cache_hit:
            session["cache_hits"] += 1
        else:
            session["cache_misses"] += 1
        
        if verification_passed:
            session["verifications_passed"] += 1
        else:
            session["verifications_failed"] += 1
        
        query_data = {
            "query": input_text[:200],
            "worker": worker_type,
            "tokens": total_tokens,
            "time_ms": latency_ms,
            "verification_passed": verification_passed,
            "cache_hit": cache_hit,
            "num_simulations": num_simulations,
            "timestamp": datetime.now().isoformat()
        }
        self.analytics.record_query(query_data)
    
    def get_session_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return {}
        
        session = self.sessions[sid]
        num_inferences = len(session["inferences"])
        
        metrics = {
            "session_id": sid,
            "total_inferences": num_inferences,
            "total_tokens": session["total_tokens"],
            "input_tokens": session["total_input_tokens"],
            "output_tokens": session["total_output_tokens"],
            "total_cost": session["total_cost"],
            "total_latency_ms": session["total_latency_ms"],
            "cache_hits": session["cache_hits"],
            "cache_misses": session["cache_misses"],
            "cache_hit_rate": (
                session["cache_hits"] / (session["cache_hits"] + session["cache_misses"])
                if (session["cache_hits"] + session["cache_misses"]) > 0 else 0
            ),
            "verifications_passed": session["verifications_passed"],
            "verifications_failed": session["verifications_failed"],
            "verification_rate": (
                session["verifications_passed"] / num_inferences if num_inferences > 0 else 0
            ),
            "avg_latency_ms": (
                session["total_latency_ms"] / num_inferences if num_inferences > 0 else 0
            ),
            "avg_tokens_per_query": (
                session["total_tokens"] / num_inferences if num_inferences > 0 else 0
            )
        }
        
        return metrics
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all sessions."""
        return self.analytics.get_summary()
    
    def export_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return "{}"
        return json.dumps(self.sessions[sid], indent=2)

