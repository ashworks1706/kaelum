#!/usr/bin/env python
"""Command-line interface for training the Neural Router (Kaelum Brain)."""

import click
import sys
from pathlib import Path

try:
    from kaelum.core.neural_router import NeuralRouter
    from kaelum.core.neural_router_trainer import NeuralRouterTrainer
    NEURAL_ROUTER_AVAILABLE = True
except ImportError as e:
    NEURAL_ROUTER_AVAILABLE = False
    print(f"⚠️  Neural router not available: {e}")
    print("   Install dependencies: pip install torch sentence-transformers")


@click.group()
def cli():
    """Neural Router (Kaelum Brain) - Training and management CLI."""
    pass


@cli.command()
@click.option('--outcomes-file', type=click.Path(exists=True), 
              help='Path to outcomes.jsonl file with historical routing data')
@click.option('--data-dir', default='.kaelum/neural_routing',
              help='Directory for model checkpoints and data')
@click.option('--epochs', default=50, type=int,
              help='Number of training epochs')
@click.option('--batch-size', default=32, type=int,
              help='Batch size for training')
@click.option('--learning-rate', default=0.001, type=float,
              help='Learning rate')
@click.option('--validation-split', default=0.2, type=float,
              help='Fraction of data for validation')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device for training (cpu or cuda)')
@click.option('--generate-synthetic', default=0, type=int,
              help='Number of synthetic samples to generate for bootstrapping')
def train(outcomes_file, data_dir, epochs, batch_size, learning_rate, 
          validation_split, device, generate_synthetic):
    """Train the neural router policy network."""
    
    if not NEURAL_ROUTER_AVAILABLE:
        click.echo("❌ Neural router dependencies not installed", err=True)
        click.echo("   Install with: pip install torch sentence-transformers", err=True)
        sys.exit(1)
    
    click.echo("=" * 70)
    click.echo("🧠 Neural Router Training")
    click.echo("=" * 70)
    
    # Initialize neural router
    neural_router = NeuralRouter(
        data_dir=data_dir,
        fallback_to_rules=True,
        device=device
    )
    
    # Initialize trainer
    trainer = NeuralRouterTrainer(
        neural_router=neural_router,
        outcomes_file=outcomes_file,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )
    
    # Generate synthetic data if requested
    if generate_synthetic > 0:
        click.echo(f"\n📝 Generating {generate_synthetic} synthetic training samples...")
        trainer.generate_synthetic_data(
            num_samples=generate_synthetic,
            save_to_outcomes=True
        )
        click.echo("✓ Synthetic data generated\n")
    
    # Train the model
    try:
        click.echo(f"\n🚀 Starting training...")
        click.echo(f"   Epochs: {epochs}")
        click.echo(f"   Batch size: {batch_size}")
        click.echo(f"   Learning rate: {learning_rate}")
        click.echo(f"   Validation split: {validation_split}")
        click.echo(f"   Device: {device}\n")
        
        history = trainer.train(
            num_epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=10,
            save_best=True
        )
        
        # Display final results
        click.echo("\n" + "=" * 70)
        click.echo("✅ Training Complete!")
        click.echo("=" * 70)
        click.echo(f"Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
        click.echo(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        click.echo(f"Model saved to: {data_dir}/neural_router.pt")
        
    except Exception as e:
        click.echo(f"\n❌ Training failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--data-dir', default='.kaelum/neural_routing',
              help='Directory for model checkpoints')
@click.option('--query', required=True,
              help='Query to route')
def test(data_dir, query):
    """Test the neural router on a query."""
    
    if not NEURAL_ROUTER_AVAILABLE:
        click.echo("❌ Neural router dependencies not installed", err=True)
        sys.exit(1)
    
    click.echo("=" * 70)
    click.echo("🧠 Neural Router Test")
    click.echo("=" * 70)
    click.echo(f"\nQuery: {query}\n")
    
    # Initialize neural router
    neural_router = NeuralRouter(
        data_dir=data_dir,
        fallback_to_rules=True,
    )
    
    # Get routing decision
    decision = neural_router.route(query)
    
    # Display results
    click.echo("\n📊 Routing Decision:")
    click.echo(f"   Query Type: {decision.query_type.value}")
    click.echo(f"   Strategy: {decision.strategy.value}")
    click.echo(f"   Max Reflection Iterations: {decision.max_reflection_iterations}")
    click.echo(f"   Use Symbolic Verification: {decision.use_symbolic_verification}")
    click.echo(f"   Use Factual Verification: {decision.use_factual_verification}")
    click.echo(f"   Confidence Threshold: {decision.confidence_threshold:.2f}")
    click.echo(f"   Complexity Score: {decision.complexity_score:.2f}")
    click.echo(f"\n   Reasoning: {decision.reasoning}")
    click.echo()


@cli.command()
@click.option('--outcomes-file', type=click.Path(exists=True),
              help='Path to outcomes.jsonl file')
def stats(outcomes_file):
    """Display statistics about routing outcomes."""
    
    if not outcomes_file:
        # Try default location
        outcomes_file = Path('.kaelum/routing/outcomes.jsonl')
        if not outcomes_file.exists():
            click.echo("❌ No outcomes file found", err=True)
            click.echo(f"   Looking for: {outcomes_file}", err=True)
            click.echo("   Specify path with --outcomes-file", err=True)
            sys.exit(1)
    else:
        outcomes_file = Path(outcomes_file)
    
    click.echo("=" * 70)
    click.echo("📊 Routing Outcomes Statistics")
    click.echo("=" * 70)
    
    import json
    from collections import Counter, defaultdict
    
    total = 0
    strategies = Counter()
    query_types = Counter()
    accuracies = []
    latencies = []
    
    with open(outcomes_file, 'r') as f:
        for line in f:
            try:
                outcome = json.loads(line)
                total += 1
                strategies[outcome.get('strategy', 'unknown')] += 1
                query_types[outcome.get('query_type', 'unknown')] += 1
                accuracies.append(outcome.get('accuracy_score', 0))
                latencies.append(outcome.get('latency_ms', 0))
            except:
                continue
    
    if total == 0:
        click.echo("\n⚠️  No outcomes found in file")
        return
    
    click.echo(f"\nTotal Outcomes: {total}")
    
    click.echo("\n📈 Strategies Used:")
    for strategy, count in strategies.most_common():
        click.echo(f"   {strategy:20s}: {count:4d} ({count/total*100:5.1f}%)")
    
    click.echo("\n🏷️  Query Types:")
    for qtype, count in query_types.most_common():
        click.echo(f"   {qtype:20s}: {count:4d} ({count/total*100:5.1f}%)")
    
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        click.echo(f"\n🎯 Average Accuracy: {avg_acc:.2%}")
    
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        click.echo(f"⚡ Average Latency: {avg_lat:.1f}ms")
    
    click.echo()


if __name__ == '__main__':
    cli()
