"""Flask API for Kaelum AI Reasoning System."""

print(">>> app.py starting - before imports", flush=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '4'

print(">>> environment vars set", flush=True)

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys
import json
import time
import logging
from pathlib import Path

print(">>> base imports done", flush=True)

print(">>> base imports done", flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(">>> path setup done", flush=True)

try:
    import torch
    print(">>> torch imported", flush=True)
    torch.set_default_device('cpu')
    if torch.cuda.is_available():
        torch.cuda.is_available = lambda: False
    print(">>> torch configured", flush=True)
except ImportError:
    print(">>> torch not available", flush=True)
    pass

print(">>> importing kaelum", flush=True)
import kaelum
print(">>> kaelum imported", flush=True)
from backend.config import DEFAULT_CONFIG, WORKER_INFO
print(">>> backend.config imported", flush=True)
from backend.logging_config import setup_backend_logging, LOG_FILE
print(">>> logging_config imported", flush=True)
from backend.metrics_utils import compute_worker_metrics, compute_verification_metrics, compute_reflection_metrics

print(">>> all imports done", flush=True)
setup_backend_logging()
print(">>> logging setup complete", flush=True)

app = Flask(__name__)
print(">>> Flask app created", flush=True)
CORS(app)
print(">>> CORS configured", flush=True)

current_config = DEFAULT_CONFIG.copy()


def initialize_kaelum():
    """Initialize Kaelum system with full configuration."""
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Initializing Kaelum AI Backend")
    logger.info(f"LLM: {current_config['model']} @ {current_config['base_url']}")
    logger.info(f"Cache Dir: {current_config['cache_dir']}")
    logger.info(f"Router Dir: {current_config['router_data_dir']}")
    logger.info(f"Active Learning: {current_config['enable_active_learning']}")
    logger.info("========================================")
    logger.info("Loading embedding model (this may take 10-30 seconds on first run)...")

    kaelum.set_reasoning_model(**current_config)
    
    logger.info("✓ Kaelum system initialized successfully")


# Lazy initialization - only initialize on first query, not on import
_initialized = False

def ensure_initialized():
    """Ensure Kaelum is initialized before processing requests."""
    global _initialized
    if not _initialized:
        initialize_kaelum()
        _initialized = True


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": kaelum.__version__, "timestamp": time.time()})


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    global current_config
    
    if request.method == 'GET':
        return jsonify(current_config)
    
    current_config.update(request.json)
    ensure_initialized()  # Initialize if needed before applying config
    kaelum.set_reasoning_model(**current_config)
    return jsonify({"status": "updated", "config": current_config})


@app.route('/api/query', methods=['POST'])
def query():
    ensure_initialized()  # Lazy initialization on first query
    data = request.json
    query_text = data.get('query', '')
    use_stream = data.get('stream', False)
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    # Clear log file at start of query
    if LOG_FILE.exists():
        with open(LOG_FILE, 'w') as f:
            f.write('')
    
    if use_stream:
        def generate():
            try:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"
                
                start_time = time.time()
                result = kaelum.kaelum_enhance_reasoning(query_text)
                execution_time = time.time() - start_time
                
                yield f"data: {json.dumps({'type': 'router', 'worker': result.get('worker_used', 'unknown'), 'confidence': result.get('confidence', 0.0)})}\n\n"
                
                for i, step in enumerate(result.get("reasoning_steps", [])):
                    yield f"data: {json.dumps({'type': 'reasoning_step', 'index': i, 'content': step})}\n\n"
                
                yield f"data: {json.dumps({'type': 'answer', 'content': result.get('suggested_approach', '')})}\n\n"
                yield f"data: {json.dumps({'type': 'verification', 'passed': result.get('verification_passed', False)})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'execution_time': execution_time, 'cache_hit': result.get('cache_hit', False), 'iterations': result.get('iterations', 1)})}\n\n"
            
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        response = Response(stream_with_context(generate()), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
    
    try:
        start_time = time.time()
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
            "execution_time": execution_time
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get logs from file with optional offset for polling."""
    limit = request.args.get('limit', type=int, default=100)
    offset = request.args.get('offset', type=int, default=0)
    logs = []
    total_lines = 0
    
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()
            total_lines = len(all_lines)
            # Apply offset and limit
            selected_lines = all_lines[offset:offset + limit] if offset > 0 else all_lines[-limit:]
            
            for line in selected_lines:
                line = line.strip()
                if line:
                    # Simple text logs, create a basic structure for frontend
                    logs.append({
                        'timestamp': '',
                        'level': 'info',
                        'logger': '',
                        'message': line
                    })
    
    return jsonify({
        "logs": logs, 
        "count": len(logs), 
        "total": total_lines,
        "timestamp": time.time()
    })


@app.route('/api/metrics', methods=['GET'])
def metrics():
    ensure_initialized()  # Lazy initialization
    try:
        metrics_data = kaelum.get_metrics()
        analytics = metrics_data.get('analytics', {})
        
        return jsonify({
            "total_queries": analytics.get('total_queries', 0),
            "total_successes": analytics.get('verified_queries', 0),
            "total_failures": analytics.get('total_queries', 0) - analytics.get('verified_queries', 0),
            "avg_execution_time": analytics.get('avg_time_ms', 0) / 1000.0,
            "avg_nodes_explored": analytics.get('avg_simulations', 0),
            "avg_iterations": 1.0,
            "cache_hit_rate": analytics.get('cache_hit_rate', 0.0),
            "worker_metrics": compute_worker_metrics(analytics),
            "verification_metrics": compute_verification_metrics(analytics),
            "reflection_metrics": compute_reflection_metrics(analytics)
        })
    except Exception:
        return jsonify({
            "total_queries": 0,
            "total_successes": 0,
            "total_failures": 0,
            "avg_execution_time": 0.0,
            "avg_nodes_explored": 0.0,
            "avg_iterations": 0.0,
            "cache_hit_rate": 0.0,
            "worker_metrics": {},
            "verification_metrics": {"total_verified": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
            "reflection_metrics": {"total_reflections": 0, "avg_iterations": 0.0, "improvement_rate": 0.0}
        })


@app.route('/api/stats/router', methods=['GET'])
def router_stats():
    try:
        # Use absolute path to root .kaelum folder
        project_root = Path(__file__).parent.parent
        router_file = project_root / ".kaelum" / "routing" / "training_data.json"
        training_data = []
        if router_file.exists():
            with open(router_file, 'r') as f:
                training_data = json.load(f)
        
        model_file = project_root / ".kaelum" / "routing" / "model.pt"
        model_trained = model_file.exists()
        training_steps = 0
        exploration_rate = current_config.get("router_exploration_rate", 0.1)
        
        if model_trained:
            try:
                import torch
                checkpoint = torch.load(model_file, map_location='cpu')
                training_steps = checkpoint.get("training_step_count", 0)
                exploration_rate = checkpoint.get("exploration_rate", exploration_rate)
            except:
                pass
        
        total_queries = len(training_data)
        workers_used = {}
        
        for entry in training_data:
            worker = entry.get('worker', 'unknown')
            workers_used[worker] = workers_used.get(worker, 0) + 1
        
        success_count = sum(1 for e in training_data if e.get('success', False))
        success_rate = success_count / total_queries if total_queries > 0 else 0.0
        buffer_size = current_config.get("router_buffer_size", 32)
        
        return jsonify({
            "total_queries": total_queries,
            "model_trained": model_trained,
            "training_steps": training_steps,
            "training_buffer_size": total_queries % buffer_size,
            "success_rate": success_rate,
            "workers_distribution": workers_used,
            "recent_queries": training_data[-5:],
            "online_learning": {
                "enabled": True,
                "learning_rate": current_config.get("router_learning_rate", 0.001),
                "buffer_size": buffer_size,
                "exploration_rate": exploration_rate,
                "training_steps": training_steps
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/cache', methods=['GET'])
def cache_stats():
    try:
        # Use absolute path to root .kaelum folder
        project_root = Path(__file__).parent.parent
        cache_file = project_root / ".kaelum" / "cache" / "metadata.json"
        validation_log = project_root / ".kaelum" / "cache_validation" / "validation_log.jsonl"
        
        cache_data = []
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        
        validation_entries = []
        if validation_log.exists():
            with open(validation_log, 'r') as f:
                for line in f:
                    try:
                        validation_entries.append(json.loads(line))
                    except:
                        pass
        
        by_worker = {}
        for entry in cache_data:
            worker = entry.get('worker_specialty', 'unknown')
            by_worker[worker] = by_worker.get(worker, 0) + 1
        
        total_validations = len(validation_entries)
        accepted = sum(1 for v in validation_entries if v.get('validation_result', {}).get('valid', False))
        
        cache_files = [{'query': e.get('query', '')[:100], 'worker': e.get('worker_specialty', 'unknown'), 
                       'nodes': e.get('num_nodes', 0), 'cache_id': e.get('cache_id', '')} 
                       for e in cache_data[:20]]
        
        return jsonify({
            "total_cached": len(cache_data),
            "by_worker": by_worker,
            "acceptance_rate": accepted / total_validations if total_validations > 0 else 0.0,
            "validations_accepted": accepted,
            "validations_rejected": total_validations - accepted,
            "total_validations": total_validations,
            "recent_validations": validation_entries[-5:],
            "cache_files": cache_files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/calibration', methods=['GET'])
def calibration_stats():
    try:
        # Use absolute path to root .kaelum folder
        project_root = Path(__file__).parent.parent
        calibration_file = project_root / ".kaelum" / "calibration" / "optimal_thresholds.json"
        decisions_file = project_root / ".kaelum" / "calibration" / "decisions.jsonl"
        
        optimal_thresholds = {}
        if calibration_file.exists():
            with open(calibration_file, 'r') as f:
                optimal_thresholds = json.load(f)
        
        decision_count = 0
        if decisions_file.exists():
            with open(decisions_file, 'r') as f:
                decision_count = sum(1 for _ in f)
        
        return jsonify({
            "optimal_thresholds": optimal_thresholds,
            "total_decisions": decision_count,
            "calibrated_tasks": list(optimal_thresholds.keys())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/training-data', methods=['GET'])
def export_training():
    ensure_initialized()  # Lazy initialization
    try:
        output_path = f"/tmp/kaelum_training_{int(time.time())}.jsonl"
        count = kaelum.export_training_data(output_path)
        return jsonify({"status": "exported", "count": count, "path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/workers', methods=['GET'])
def list_workers():
    return jsonify({"workers": WORKER_INFO})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 KAELUM API SERVER STARTING...")
    print("="*70)
    print(f"⚙️  CPU-ONLY MODE: GPU reserved for vLLM")
    print(f"🌐 API: http://localhost:5000")
    print(f"💚 Health: http://localhost:5000/api/health")
    print(f"📊 Metrics: http://localhost:5000/api/metrics")
    print("="*70)
    print("📝 NOTE: First query will take 10-30s to load embedding model")
    print("="*70 + "\n")
    
    logger = logging.getLogger(__name__)
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    
