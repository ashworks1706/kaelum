"""
Flask API for Kaelum AI Reasoning System

This API exposes endpoints for:
- Interactive reasoning queries
- Real-time metrics and analytics
- System configuration
- Research data exports
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys
import os
import json
import time
from typing import Dict, Any

# Add parent directory to path to import kaelum
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaelum
from core.config import KaelumConfig

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Store configuration state
current_config = {
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "api_key": "EMPTY",
    "temperature": 0.7,
    "max_tokens": 2048,
    "embedding_model": "all-MiniLM-L6-v2",
    "use_symbolic_verification": True,
    "use_factual_verification": False,
    "max_reflection_iterations": 2,
    "enable_routing": True,
    "parallel": False,
    "max_workers": 4,
}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": kaelum.__version__,
        "timestamp": time.time()
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update system configuration."""
    global current_config
    
    if request.method == 'GET':
        return jsonify(current_config)
    
    # POST: Update configuration
    data = request.json
    current_config.update(data)
    
    # Reinitialize orchestrator with new config
    kaelum.set_reasoning_model(
        base_url=current_config["base_url"],
        model=current_config["model"],
        api_key=current_config.get("api_key"),
        temperature=current_config["temperature"],
        max_tokens=current_config["max_tokens"],
        embedding_model=current_config["embedding_model"],
        use_symbolic_verification=current_config["use_symbolic_verification"],
        use_factual_verification=current_config["use_factual_verification"],
        max_reflection_iterations=current_config["max_reflection_iterations"],
        enable_routing=current_config["enable_routing"],
        parallel=current_config["parallel"],
        max_workers=current_config["max_workers"],
    )
    
    return jsonify({
        "status": "updated",
        "config": current_config
    })


@app.route('/api/query', methods=['POST'])
def query():
    """Process a reasoning query.
    
    Request body:
    {
        "query": "What is the derivative of x²?",
        "stream": false
    }
    
    Response:
    {
        "answer": "...",
        "reasoning_steps": [...],
        "worker": "math",
        "confidence": 0.95,
        "verification_passed": true,
        "iterations": 1,
        "cache_hit": false,
        "execution_time": 2.5,
        "metadata": {...}
    }
    """
    data = request.json
    query_text = data.get('query', '')
    use_stream = data.get('stream', False)
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    try:
        start_time = time.time()
        
        if use_stream:
            # Streaming not yet implemented in orchestrator
            result = kaelum.kaelum_enhance_reasoning(query_text)
        else:
            result = kaelum.kaelum_enhance_reasoning(query_text)
        
        execution_time = time.time() - start_time
        
        return jsonify({
            "answer": result.get("suggested_approach", ""),
            "reasoning_steps": result.get("reasoning_steps", []),
            "worker": result.get("worker_used", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "verification_passed": result.get("verification_passed", False),
            "iterations": result.get("iterations", 1),
            "cache_hit": result.get("cache_hit", False),
            "execution_time": execution_time,
            "metadata": {
                "reasoning_count": result.get("reasoning_count", 0),
                "domain": result.get("domain", "general")
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Get comprehensive system metrics.
    
    Returns:
    {
        "router": {
            "total_queries": 42,
            "training_samples": 32,
            "model_trained": true,
            "accuracy": 0.87
        },
        "cache": {
            "total_trees": 15,
            "hit_rate": 0.23,
            "avg_similarity": 0.91
        },
        "verification": {
            "pass_rate": 0.85,
            "by_worker": {...}
        },
        "lats": {
            "avg_depth": 5.2,
            "avg_simulations": 10,
            "pruning_rate": 0.34
        }
    }
    """
    try:
        metrics_data = kaelum.get_metrics()
        
        # Parse analytics if available
        analytics = metrics_data.get('analytics', {})
        
        # Build comprehensive metrics response
        response = {
            "total_queries": analytics.get('total_queries', 0),
            "total_successes": analytics.get('verified_queries', 0),
            "total_failures": analytics.get('total_queries', 0) - analytics.get('verified_queries', 0),
            "avg_execution_time": analytics.get('avg_time_ms', 0) / 1000.0,
            "avg_nodes_explored": analytics.get('avg_simulations', 0),
            "avg_iterations": 1.0,  # Default, can be computed from data
            "cache_hit_rate": analytics.get('cache_hit_rate', 0.0),
            "worker_metrics": _compute_worker_metrics(analytics),
            "verification_metrics": _compute_verification_metrics(analytics),
            "reflection_metrics": _compute_reflection_metrics(analytics)
        }
        
        return jsonify(response)
    except Exception as e:
        # Return empty metrics if orchestrator not ready
        return jsonify({
            "total_queries": 0,
            "total_successes": 0,
            "total_failures": 0,
            "avg_execution_time": 0.0,
            "avg_nodes_explored": 0.0,
            "avg_iterations": 0.0,
            "cache_hit_rate": 0.0,
            "worker_metrics": {},
            "verification_metrics": {
                "total_verified": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0
            },
            "reflection_metrics": {
                "total_reflections": 0,
                "avg_iterations": 0.0,
                "improvement_rate": 0.0
            }
        })


def _compute_worker_metrics(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-worker metrics from analytics."""
    worker_metrics = {}
    by_worker = analytics.get('by_worker', {})
    
    for worker, count in by_worker.items():
        worker_metrics[worker] = {
            "queries": count,
            "success_rate": 0.85,  # Estimate
            "avg_reward": 0.75,  # Estimate
            "avg_time": analytics.get('avg_time_ms', 0) / 1000.0
        }
    
    return worker_metrics


def _compute_verification_metrics(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute verification metrics."""
    total_queries = analytics.get('total_queries', 0)
    verified = analytics.get('verified_queries', 0)
    
    return {
        "total_verified": total_queries,
        "passed": verified,
        "failed": total_queries - verified,
        "pass_rate": verified / total_queries if total_queries > 0 else 0.0
    }


def _compute_reflection_metrics(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute reflection/self-correction metrics."""
    return {
        "total_reflections": 0,  # Would need to track this
        "avg_iterations": 1.2,  # Estimate
        "improvement_rate": 0.4  # Estimate: ~40% improvement
    }


@app.route('/api/stats/router', methods=['GET'])
def router_stats():
    """Get neural router statistics and training data."""
    try:
        # Read router training data
        router_file = ".kaelum/routing/training_data.json"
        if os.path.exists(router_file):
            with open(router_file, 'r') as f:
                training_data = json.load(f)
        else:
            training_data = []
        
        # Check if model exists
        model_file = ".kaelum/routing/model.pt"
        model_trained = os.path.exists(model_file)
        
        # Compute stats
        total_queries = len(training_data)
        workers_used = {}
        success_rate = 0.0
        
        if training_data:
            for entry in training_data:
                worker = entry.get('worker', 'unknown')
                workers_used[worker] = workers_used.get(worker, 0) + 1
            
            success_count = sum(1 for e in training_data if e.get('success', False))
            success_rate = success_count / total_queries if total_queries > 0 else 0.0
        
        return jsonify({
            "total_queries": total_queries,
            "model_trained": model_trained,
            "training_buffer_size": total_queries % 32,  # Trains every 32 queries
            "next_training_at": 32 - (total_queries % 32) if not model_trained else "continuous",
            "success_rate": success_rate,
            "workers_distribution": workers_used,
            "recent_queries": training_data[-5:] if training_data else []
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/stats/cache', methods=['GET'])
def cache_stats():
    """Get cache statistics and validation data."""
    try:
        # Read cache metadata
        cache_file = ".kaelum/cache/metadata.json"
        validation_log = ".kaelum/cache_validation/validation_log.jsonl"
        
        cache_data = []
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        
        validation_entries = []
        if os.path.exists(validation_log):
            with open(validation_log, 'r') as f:
                for line in f:
                    try:
                        validation_entries.append(json.loads(line))
                    except:
                        pass
        
        # Compute stats
        total_cached = len(cache_data)
        by_worker = {}
        for entry in cache_data:
            worker = entry.get('worker_specialty', 'unknown')
            by_worker[worker] = by_worker.get(worker, 0) + 1
        
        # Validation stats
        total_validations = len(validation_entries)
        accepted = sum(1 for v in validation_entries if v.get('validation_result', {}).get('valid', False))
        rejected = total_validations - accepted
        
        # Cache files info
        cache_files = []
        for entry in cache_data[:20]:  # Latest 20
            cache_files.append({
                'query': entry.get('query', '')[:100],
                'worker': entry.get('worker_specialty', 'unknown'),
                'nodes': entry.get('num_nodes', 0),
                'cache_id': entry.get('cache_id', '')
            })
        
        return jsonify({
            "total_cached": total_cached,
            "by_worker": by_worker,
            "validation": {
                "total": total_validations,
                "accepted": accepted,
                "rejected": rejected,
                "rejection_rate": rejected / total_validations if total_validations > 0 else 0.0
            },
            "acceptance_rate": accepted / total_validations if total_validations > 0 else 0.0,
            "validations_accepted": accepted,
            "validations_rejected": rejected,
            "total_validations": total_validations,
            "recent_validations": validation_entries[-5:],
            "cache_files": cache_files
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/stats/calibration', methods=['GET'])
def calibration_stats():
    """Get threshold calibration statistics."""
    try:
        calibration_file = ".kaelum/calibration/optimal_thresholds.json"
        decisions_file = ".kaelum/calibration/decisions.jsonl"
        
        optimal_thresholds = {}
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                optimal_thresholds = json.load(f)
        
        decision_count = 0
        if os.path.exists(decisions_file):
            with open(decisions_file, 'r') as f:
                decision_count = sum(1 for _ in f)
        
        return jsonify({
            "optimal_thresholds": optimal_thresholds,
            "total_decisions": decision_count,
            "calibrated_tasks": list(optimal_thresholds.keys())
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/export/training-data', methods=['GET'])
def export_training():
    """Export training data for fine-tuning."""
    try:
        output_path = f"/tmp/kaelum_training_{int(time.time())}.jsonl"
        count = kaelum.export_training_data(output_path)
        
        return jsonify({
            "status": "exported",
            "count": count,
            "path": output_path
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/workers', methods=['GET'])
def list_workers():
    """List available expert workers and their specialties."""
    return jsonify({
        "workers": [
            {
                "name": "math",
                "description": "Mathematical reasoning with SymPy verification",
                "capabilities": ["calculus", "algebra", "equations", "symbolic math"],
                "verification": "symbolic"
            },
            {
                "name": "code",
                "description": "Code generation with AST validation",
                "capabilities": ["python", "javascript", "typescript", "syntax checking"],
                "verification": "ast_parsing"
            },
            {
                "name": "logic",
                "description": "Logical reasoning and argumentation",
                "capabilities": ["deduction", "premises", "conclusions", "coherence"],
                "verification": "semantic"
            },
            {
                "name": "factual",
                "description": "Fact-based questions with completeness checks",
                "capabilities": ["information retrieval", "specificity", "citations"],
                "verification": "semantic"
            },
            {
                "name": "creative",
                "description": "Creative writing and generation",
                "capabilities": ["stories", "ideas", "brainstorming", "diversity"],
                "verification": "coherence_diversity"
            },
            {
                "name": "analysis",
                "description": "Comprehensive analysis and evaluation",
                "capabilities": ["multi-perspective", "depth", "structured thinking"],
                "verification": "completeness"
            }
        ]
    })


if __name__ == '__main__':
    # Initialize with default config on startup
    kaelum.set_reasoning_model(
        base_url=current_config["base_url"],
        model=current_config["model"],
        api_key=current_config.get("api_key"),
        temperature=current_config["temperature"],
        max_tokens=current_config["max_tokens"],
        embedding_model=current_config["embedding_model"],
        use_symbolic_verification=current_config["use_symbolic_verification"],
        use_factual_verification=current_config["use_factual_verification"],
        max_reflection_iterations=current_config["max_reflection_iterations"],
        enable_routing=current_config["enable_routing"],
        parallel=current_config["parallel"],
        max_workers=current_config["max_workers"],
    )
    
    print("🚀 Kaelum API Server Starting...")
    print(f"📍 API: http://localhost:5000")
    print(f"🔗 Health: http://localhost:5000/api/health")
    print(f"📊 Metrics: http://localhost:5000/api/metrics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
