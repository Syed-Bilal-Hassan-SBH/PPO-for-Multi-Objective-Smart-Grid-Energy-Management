#!/usr/bin/env python3
"""
Process Multi-Seed Results
Extracts and aggregates results from multiple seed runs for analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

def extract_final_returns(metrics_data: Dict) -> float:
    """Extract final episode returns from metrics data."""
    # Look for episode return data
    if 'train/episode_return' in metrics_data:
        returns_data = metrics_data['train/episode_return']
        if returns_data and len(returns_data) > 0:
            # Take the last 10 episodes and average them
            final_returns = [x[1] for x in returns_data[-10:]]
            return np.mean(final_returns)
    
    # Fallback: look for any return-related metric
    for key in metrics_data.keys():
        if 'return' in key.lower():
            data = metrics_data[key]
            if data and len(data) > 0:
                values = [x[1] if isinstance(x, list) else x for x in data[-10:]]
                return np.mean(values)
    
    # Default fallback
    return -1000.0

def extract_training_metrics(metrics_data: Dict) -> Dict:
    """Extract key training metrics."""
    metrics = {}
    
    # Extract final losses
    for key in ['train/policy_loss', 'train/value_loss', 'train/entropy']:
        if key in metrics_data:
            data = metrics_data[key]
            if data and len(data) > 0:
                # Take last 5 values and average
                final_values = [x[1] for x in data[-5:]]
                metrics[key] = np.mean(final_values)
    
    # Extract KL divergence
    if 'train/kl_divergence' in metrics_data:
        data = metrics_data['train/kl_divergence']
        if data and len(data) > 0:
            final_values = [x[1] for x in data[-5:]]
            metrics['kl_divergence'] = np.mean(final_values)
    
    return metrics

def process_seed_results(results_dir: str, n_seeds: int = 5) -> Dict:
    """Process results from multiple seeds."""
    results_dir = Path(results_dir)
    seed_results = {}
    
    for seed in range(n_seeds):
        seed_dir = results_dir / f"seed_{seed}"
        
        # Find metrics files
        metrics_files = list(seed_dir.rglob("metrics.json"))
        
        if not metrics_files:
            print(f"Warning: No metrics found for seed {seed}")
            continue
        
        # Use the most recent metrics file
        metrics_file = sorted(metrics_files, key=lambda p: p.stat().st_mtime)[-1]
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # Extract key metrics
        final_return = extract_final_returns(metrics_data)
        training_metrics = extract_training_metrics(metrics_data)
        
        seed_results[seed] = {
            'mean_return': final_return,
            **training_metrics
        }
        
        print(f"Seed {seed}: Return = {final_return:.2f}")
    
    return seed_results

def create_multi_seed_plots(seed_results: Dict, output_dir: str):
    """Create plots from multi-seed results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract returns
    returns = [seed_results[seed]['mean_return'] for seed in seed_results.keys()]
    seeds = list(seed_results.keys())
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Returns per seed
    axes[0, 0].bar(seeds, returns, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Seed')
    axes[0, 0].set_ylabel('Final Return')
    axes[0, 0].set_title('Final Returns by Seed')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Return distribution
    axes[0, 1].hist(returns, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Final Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Return Distribution Across Seeds')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics summary
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    ci_lower = np.percentile(returns, 2.5)
    ci_upper = np.percentile(returns, 97.5)
    
    stats_text = f"""Multi-Seed Statistics:
    
Mean Return: {mean_return:.2f} ± {std_return:.2f}
95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]
Min: {np.min(returns):.2f}
Max: {np.max(returns):.2f}
N Seeds: {len(returns)}"""
    
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Learning curves (if available)
    if len(seed_results) > 0:
        sample_seed = list(seed_results.keys())[0]
        # This would need the full training curves - for now just show summary
        axes[1, 1].text(0.1, 0.5, f"Sample Seed: {sample_seed}\nFinal Return: {seed_results[sample_seed]['mean_return']:.2f}", 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved multi-seed analysis plot to: {output_dir / 'multi_seed_analysis.png'}")
    
    # Save summary statistics
    summary = {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'se_return': float(std_return / np.sqrt(len(returns))),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'n_seeds': len(returns),
        'individual_returns': returns
    }
    
    with open(output_dir / "multi_seed_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved multi-seed summary to: {output_dir / 'multi_seed_summary.json'}")
    
    return summary

def main():
    """Main processing function."""
    print("="*60)
    print("PROCESSING MULTI-SEED RESULTS")
    print("="*60)
    
    # Process seed results
    seed_results = process_seed_results("results/multi_seed", n_seeds=5)
    
    if len(seed_results) == 0:
        print("No seed results found!")
        return
    
    print(f"\nProcessed {len(seed_results)} seeds")
    
    # Create plots and summary
    summary = create_multi_seed_plots(seed_results, "results/final_results")
    
    print("\n" + "="*60)
    print("MULTI-SEED ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Mean Return: {summary['mean_return']:.2f} ± {summary['se_return']:.2f}")
    print(f"95% CI: [{summary['ci_lower']:.2f}, {summary['ci_upper']:.2f}]")
    print("="*60)

if __name__ == "__main__":
    main()
