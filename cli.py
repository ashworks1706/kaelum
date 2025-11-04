"""Kaelum CLI for model management and testing."""

import click
import sys
from pathlib import Path


@click.group()
@click.version_option(version="1.5.0")
def cli():
    """Kaelum CLI - Local reasoning models as cognitive middleware."""
    pass


@cli.command()
@click.option('--model', default="Qwen/Qwen2.5-7B-Instruct", help='Model name')
@click.option('--port', default=8000, help='Port for vLLM server')
@click.option('--gpu-memory', default=0.9, help='GPU memory utilization')
def serve(model, port, gpu_memory):
    """Start vLLM server with recommended settings."""
    click.echo(f"🚀 Starting vLLM server with {model}...")
    
    import subprocess
    
    cmd = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model,
        '--port', str(port),
        '--gpu-memory-utilization', str(gpu_memory),
        '--max-model-len', '2048'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n⏹️  Server stopped")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--model', default="http://localhost:8000/v1", help='Model API endpoint')
@click.option('--stream/--no-stream', default=False, help='Stream output')
@click.option('--debug/--no-debug', default=False, help='Debug verification')
def query(query, model, stream, debug):
    """Run a reasoning query through Kaelum."""
    from kaelum import set_reasoning_model, enhance, enhance_stream
    
    click.echo(f"🧠 Processing query with Kaelum...\n")
    
    set_reasoning_model(
        base_url=model,
        use_symbolic_verification=True,
        debug_verification=debug
    )
    
    try:
        if stream:
            for chunk in enhance_stream(query):
                click.echo(chunk, nl=False)
            click.echo()
        else:
            result = enhance(query)
            click.echo(result)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', default='benchmark_results.json', help='Output file')
def benchmark(output):
    """Run GSM8K-style math benchmark."""
    click.echo("📊 Running benchmark...")
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from benchmarks.gsm8k_benchmark import run_benchmark, calculate_metrics, print_summary, save_results
    
    try:
        results = run_benchmark(use_kaelum=True)
        metrics = calculate_metrics(results)
        print_summary(metrics)
        save_results(results, metrics, output)
        
        if metrics['accuracy'] >= 80:
            click.echo(f"✅ Benchmark passed with {metrics['accuracy']:.1f}% accuracy")
        else:
            click.echo(f"⚠️  Benchmark completed with {metrics['accuracy']:.1f}% accuracy")
            
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def test():
    """Run test suite."""
    import subprocess
    
    click.echo("🧪 Running tests...\n")
    
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/', '-v'],
            check=False,
            capture_output=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table')
def models(format):
    """List available models."""
    from kaelum.core.registry import get_registry
    
    registry = get_registry()
    all_models = registry.list_all()
    
    if not all_models:
        click.echo("No models registered")
        return
    
    if format == 'json':
        import json
        from dataclasses import asdict
        click.echo(json.dumps([asdict(m) for m in all_models], indent=2))
    else:
        click.echo("\n📋 Registered Models:\n")
        for model in all_models:
            click.echo(f"  • {model.model_id} ({model.model_type})")
            click.echo(f"    {model.description}")
            click.echo()


@cli.command()
def health():
    """Check system health and dependencies."""
    click.echo("🏥 Checking Kaelum health...\n")
    
    # Check Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"✓ Python {py_version}")
    
    # Check dependencies
    deps = [
        ('pydantic', 'Pydantic'),
        ('sympy', 'SymPy'),
        ('httpx', 'HTTPX'),
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            click.echo(f"✓ {name} installed")
        except ImportError:
            click.echo(f"✗ {name} missing", err=True)
    
    # Check Kaelum import
    try:
        import kaelum
        click.echo(f"✓ Kaelum v{getattr(kaelum, '__version__', '1.5.0')} imported")
    except ImportError as e:
        click.echo(f"✗ Kaelum import failed: {e}", err=True)
    
    click.echo("\n✅ Health check complete")


@cli.command()
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--export', type=click.Path(), help='Export metrics to file')
def routing_stats(format, export):
    """Display routing performance statistics."""
    from kaelum.core.router_metrics import RouterMetricsCollector
    
    click.echo("📊 Collecting routing statistics...\n")
    
    try:
        collector = RouterMetricsCollector()
        metrics = collector.collect_metrics()
        
        if metrics['total_queries'] == 0:
            click.echo("⚠️  No routing data available yet.")
            click.echo("   Run some queries with routing enabled to see statistics.")
            return
        
        if format == 'json':
            import json
            output = json.dumps(metrics, indent=2)
            click.echo(output)
        else:
            # Text format with colors
            summary = collector.format_summary(metrics)
            
            # Add colors for better readability
            summary = summary.replace("ROUTER PERFORMANCE SUMMARY", 
                                    click.style("ROUTER PERFORMANCE SUMMARY", fg='green', bold=True))
            summary = summary.replace("BY STRATEGY", 
                                    click.style("BY STRATEGY", fg='cyan', bold=True))
            summary = summary.replace("BY QUERY TYPE", 
                                    click.style("BY QUERY TYPE", fg='cyan', bold=True))
            
            click.echo(summary)
            
            # Show top performing strategies
            top_strategies = collector.get_top_strategies(n=3)
            if top_strategies:
                click.echo("\n" + click.style("🏆 TOP PERFORMING STRATEGIES:", fg='yellow', bold=True))
                for i, strategy in enumerate(top_strategies, 1):
                    click.echo(f"  {i}. {strategy}")
        
        # Export if requested
        if export:
            import json
            with open(export, 'w') as f:
                json.dump(metrics, f, indent=2)
            click.echo(f"\n✅ Metrics exported to {export}")
            
    except Exception as e:
        click.echo(f"❌ Error collecting metrics: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()
