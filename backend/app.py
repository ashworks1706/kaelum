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
                
                # Safely extract values with proper defaults
                worker_used = result.get('worker_used') or 'unknown'
                confidence = result.get('confidence') or 0.0
                reasoning_steps = result.get("reasoning_steps") or []
                suggested_approach = result.get('suggested_approach') or ''
                verification_passed = result.get('verification_passed') or False
                cache_hit = result.get('cache_hit') or False
                iterations = result.get('iterations') or 1
                
                yield f"data: {json.dumps({'type': 'router', 'worker': worker_used, 'confidence': confidence})}\n\n"
                
                for i, step in enumerate(reasoning_steps):
                    if step:  # Only yield if step is not None
                        yield f"data: {json.dumps({'type': 'reasoning_step', 'index': i, 'content': step})}\n\n"
                
                yield f"data: {json.dumps({'type': 'answer', 'content': suggested_approach})}\n\n"
                yield f"data: {json.dumps({'type': 'verification', 'passed': verification_passed})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'execution_time': execution_time, 'cache_hit': cache_hit, 'iterations': iterations})}\n\n"
            
            except Exception as e:
                logger.error(f"Error in streaming query: {e}", exc_info=True)
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
        
        # Get actual router training buffer size if available
        actual_buffer_size = total_queries % buffer_size if total_queries > 0 else 0
        
        # Try to get the actual router's training buffer size from orchestrator
        try:
            metrics_data = kaelum.get_metrics()
            router_info = metrics_data.get('analytics', {}).get('router', {})
            if 'training_buffer_size' in router_info:
                actual_buffer_size = router_info['training_buffer_size']
        except:
            pass
        
        return jsonify({
            "total_queries": total_queries,
            "model_trained": model_trained,
            "training_steps": training_steps,
            "training_buffer_size": actual_buffer_size,
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
            # Handle JSONL format (one JSON object per line)
            with open(cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            cache_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        
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
        rejected = total_validations - accepted
        
        # Load node counts from actual tree files
        cache_files = []
        trees_dir = project_root / ".kaelum" / "cache" / "trees"
        for e in cache_data[:20]:
            tree_path = e.get('tree_path', '')
            node_count = 0
            
            # Try to load the tree file to get node count
            if tree_path and Path(tree_path).exists():
                try:
                    with open(tree_path, 'r') as tf:
                        tree_data = json.load(tf)
                        node_count = tree_data.get('tree_stats', {}).get('total_nodes', 0)
                except:
                    pass
            
            cache_files.append({
                'query': e.get('query', '')[:100], 
                'worker': e.get('worker_specialty', 'unknown'), 
                'nodes': node_count,
                'cache_id': e.get('tree_id', '')
            })
        
        # Calculate tree-specific stats
        high_quality = sum(1 for e in cache_data if e.get('confidence', 0) > 0.7 or e.get('success', False))
        low_quality = len(cache_data) - high_quality
        
        # Calculate cache hit rate from metadata
        cache_hits = sum(1 for e in cache_data if e.get('cache_hit', False))
        hit_rate = cache_hits / len(cache_data) if len(cache_data) > 0 else 0.0
        
        return jsonify({
            "total_cached": len(cache_data),
            "by_worker": by_worker,
            "acceptance_rate": accepted / total_validations if total_validations > 0 else 0.0,
            "rejection_rate": rejected / total_validations if total_validations > 0 else 0.0,
            "validations_accepted": accepted,
            "validations_rejected": rejected,
            "total_validations": total_validations,
            "recent_validations": validation_entries[-5:],
            "cache_files": cache_files,
            # Tree-specific stats for TreesVisualization
            "total_trees": len(cache_data),
            "high_quality": high_quality,
            "low_quality": low_quality,
            "hit_rate": hit_rate,
            "avg_similarity": 0.85  # Default value, can be calculated from actual similarity scores
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
def get_workers():
    """Get list of available workers"""
    return jsonify({
        'workers': ['math', 'logic', 'code', 'factual', 'creative', 'analysis']
    })


@app.route('/api/trees', methods=['GET'])
def get_reasoning_trees():
    """Get all reasoning trees from cache for visualization"""
    try:
        import json
        from pathlib import Path
        
        # Trees are stored in .kaelum/cache/trees/
        project_root = Path(__file__).parent.parent
        trees_dir = project_root / ".kaelum" / "cache" / "trees"
        trees = []
        
        if not trees_dir.exists():
            return jsonify({'trees': [], 'total': 0})
        
        # Read all tree files
        for tree_file in sorted(trees_dir.glob('tree_*.json'), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)
                    
                    # Extract result and tree data
                    result = tree_data.get('result', {})
                    metrics = result.get('metrics', {})
                    lats_tree = tree_data.get('lats_tree', None)
                    tree_stats = tree_data.get('tree_stats', {})
                    
                    # If we have the full LATS tree structure, use it
                    if lats_tree:
                        root_node = parse_lats_node(lats_tree)
                        best_path = extract_best_path(lats_tree)
                        mark_best_path(root_node, best_path)  # Mark nodes on best path
                        total_nodes = tree_stats.get('total_nodes', count_nodes(lats_tree))
                        pruned_nodes = count_pruned_nodes(lats_tree)
                        max_depth = tree_stats.get('max_depth', metrics.get('tree_depth', 1))
                        avg_reward = tree_stats.get('avg_reward', result.get('confidence', 0.0))
                    else:
                        # Fallback to simple single-node structure for old cache entries
                        root_node = {
                            'id': 'root',
                            'query': result.get('query', ''),
                            'children': [],
                            'visits': metrics.get('num_simulations', 0),
                            'total_reward': result.get('confidence', 0.0),
                            'avg_reward': result.get('confidence', 0.0),
                            'is_pruned': False,
                            'is_best_path': True,
                            'depth': 0,
                            'worker_type': result.get('worker', 'unknown')
                        }
                        best_path = ['root']
                        total_nodes = metrics.get('num_simulations', 0)
                        pruned_nodes = 0
                        max_depth = metrics.get('tree_depth', 1)
                        avg_reward = result.get('confidence', 0.0)
                    
                    # Convert to format expected by frontend
                    tree_info = {
                        'tree_id': tree_file.stem,
                        'query': result.get('query', 'Unknown query'),
                        'worker': result.get('worker', tree_data.get('worker', 'unknown')),
                        'timestamp': tree_file.stat().st_mtime,
                        'root': root_node,
                        'best_path': best_path,
                        'total_nodes': total_nodes,
                        'pruned_nodes': pruned_nodes,
                        'max_depth': max_depth,
                        'avg_reward': avg_reward,
                        'cache_status': tree_data.get('quality', 'none'),
                        'execution_time': metrics.get('execution_time_ms', 0) / 1000.0,
                        'verification_passed': result.get('verification_passed', False)
                    }
                    trees.append(tree_info)
            except Exception as e:
                print(f"Error reading tree file {tree_file}: {e}")
                continue
        
        return jsonify({
            'trees': trees,
            'total': len(trees)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trees': []}), 500


def parse_lats_node(node_dict):
    """Convert LATS node dict to frontend format"""
    visits = node_dict.get('visits', 0)
    value = node_dict.get('value', 0.0)
    avg_reward = value / visits if visits > 0 else 0.0
    
    return {
        'id': node_dict.get('id', 'unknown'),
        'query': node_dict.get('state', {}).get('step', node_dict.get('state', {}).get('query', '')),
        'children': [parse_lats_node(child) for child in node_dict.get('children', [])],
        'visits': visits,
        'total_reward': value,
        'avg_reward': avg_reward,
        'is_pruned': node_dict.get('pruned', False),
        'is_best_path': False,  # Will be marked later based on best_path
        'depth': node_dict.get('state', {}).get('depth', 0),
        'worker_type': node_dict.get('state', {}).get('worker', 'unknown')
    }


def extract_best_path(tree_dict):
    """Extract the best path through the tree"""
    path = []
    current = tree_dict
    
    while current:
        path.append(current.get('id', 'unknown'))
        children = current.get('children', [])
        
        if not children:
            break
        
        # Find child with highest average reward
        best_child = max(children, key=lambda c: c.get('value', 0) / max(1, c.get('visits', 1)))
        current = best_child
    
    return path


def count_nodes(tree_dict):
    """Count total nodes in tree"""
    count = 1
    for child in tree_dict.get('children', []):
        count += count_nodes(child)
    return count


def count_pruned_nodes(tree_dict):
    """Count pruned nodes in tree"""
    count = 1 if tree_dict.get('pruned', False) else 0
    for child in tree_dict.get('children', []):
        count += count_pruned_nodes(child)
    return count


def mark_best_path(node, best_path_ids):
    """Mark nodes that are on the best path"""
    if node['id'] in best_path_ids:
        node['is_best_path'] = True
    
    for child in node.get('children', []):
        mark_best_path(child, best_path_ids)


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit human feedback to improve system performance."""
    try:
        data = request.json
        
        # Import feedback engine
        from core.learning.human_feedback import HumanFeedbackEngine, HumanFeedback
        import hashlib
        
        # Create feedback object
        query = data.get('query', '')
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        feedback = HumanFeedback(
            query=query,
            query_hash=query_hash,
            timestamp=time.time(),
            overall_liked=data.get('overall_liked', False),
            overall_rating=data.get('overall_rating', 3),
            worker_selected=data.get('worker_selected', ''),
            worker_correct=data.get('worker_correct', True),
            answer_correct=data.get('answer_correct', True),
            answer_helpful=data.get('answer_helpful', True),
            answer_complete=data.get('answer_complete', True),
            answer_rating=data.get('answer_rating', 3),
            confidence_shown=data.get('confidence_shown', 0.0),
            verification_passed=data.get('verification_passed', False),
            execution_time=data.get('execution_time', 0.0),
            suggested_worker=data.get('suggested_worker'),
            steps_helpful=data.get('steps_helpful', []),
            steps_rating=data.get('steps_rating', []),
            comment=data.get('comment')
        )
        
        # Submit feedback
        engine = HumanFeedbackEngine()
        result = engine.submit_feedback(feedback)
        
        logger.info(f"✅ FEEDBACK: Received and processed feedback for query: {query[:50]}...")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ FEEDBACK: Error processing feedback: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get aggregated feedback statistics."""
    try:
        from core.learning.human_feedback import HumanFeedbackEngine
        
        engine = HumanFeedbackEngine()
        stats = engine.get_statistics()
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"❌ FEEDBACK STATS: Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/worker/<worker>', methods=['GET'])
def get_worker_feedback(worker):
    """Get feedback statistics for a specific worker."""
    try:
        from core.learning.human_feedback import HumanFeedbackEngine
        
        engine = HumanFeedbackEngine()
        stats = engine.get_worker_performance(worker)
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"❌ WORKER FEEDBACK: Error getting worker stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/router-impact', methods=['GET'])
def get_router_feedback_impact():
    """Get current impact of human feedback on router decisions."""
    try:
        ensure_initialized()
        
        # Get feedback-enhanced stats from orchestrator
        metrics = kaelum.get_metrics()
        feedback_data = metrics.get('human_feedback', {})
        
        return jsonify({
            "worker_adjustments": feedback_data.get('worker_reward_adjustments', {}),
            "step_quality_multiplier": feedback_data.get('step_quality_multiplier', 1.0),
            "worker_performance": feedback_data.get('worker_feedback_performance', {}),
            "total_feedback": feedback_data.get('total_feedback_count', 0),
            "statistics": feedback_data.get('feedback_statistics', {})
        })
    
    except Exception as e:
        logger.error(f"❌ ROUTER FEEDBACK IMPACT: Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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
    
