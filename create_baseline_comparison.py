#!/usr/bin/env python3
"""
Create Baseline Comparison Plots
Generates comparison plots between PPO and baseline methods.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_multi_seed_summary():
    """Load multi-seed PPO results."""
    summary_file = Path("results/final_results/multi_seed_summary.json")
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None

def load_baseline_results():
    """Load baseline evaluation results."""
    baseline_file = Path("results/baselines/baseline_comparison_seed42.json")
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            return json.load(f)
    return None

def create_comparison_plot(ppo_data, baseline_data, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    ppo_mean = ppo_data['mean_return']
    ppo_se = ppo_data['se_return']
    ppo_ci_lower = ppo_data['ci_lower']
    ppo_ci_upper = ppo_data['ci_upper']
    
    baselines = baseline_data['results']
    baseline_names = list(baselines.keys())
    baseline_means = [baselines[name]['mean_return'] for name in baseline_names]
    baseline_stds = [baselines[name]['std_return'] for name in baseline_names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance Comparison Bar Chart
    methods = ['PPO (Ours)'] + baseline_names
    means = [ppo_mean] + baseline_means
    stds = [ppo_se] + baseline_stds
    
    colors = ['#2ecc71'] + ['#95a5a6'] * len(baseline_names)
    bars = axes[0, 0].bar(methods, means, yerr=stds, capsize=5, 
                          color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight PPO
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(3)
    
    axes[0, 0].set_xlabel('Method', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Return', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Performance Comparison (Mean ± SE)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Improvement Over Best Baseline
    best_baseline_mean = max(baseline_means)
    improvements = [(mean - best_baseline_mean) / abs(best_baseline_mean) * 100 
                    for mean in means]
    
    colors_improvement = ['green' if imp > 0 else 'red' for imp in improvements]
    bars2 = axes[0, 1].bar(methods, improvements, color=colors_improvement, alpha=0.7)
    
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Method', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Improvement over Best Baseline ({best_baseline_mean:.0f})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Confidence Intervals
    methods_ci = ['PPO (Ours)'] + baseline_names
    ci_lower = [ppo_ci_lower] + baseline_means  # Baselines have no CI, use mean
    ci_upper = [ppo_ci_upper] + baseline_means
    
    # Create error bar plot for confidence intervals
    x_pos = np.arange(len(methods_ci))
    axes[1, 0].errorbar(x_pos, means, 
                       yerr=[np.array(means) - np.array(ci_lower), 
                             np.array(ci_upper) - np.array(means)],
                       fmt='o', capsize=10, capthick=2, markersize=8)
    
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods_ci, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Mean Return', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary Statistics Table
    stats_text = "PERFORMANCE SUMMARY\n" + "="*30 + "\n\n"
    stats_text += f"PPO (Ours):\n"
    stats_text += f"  Mean: {ppo_mean:.2f} ± {ppo_se:.2f}\n"
    stats_text += f"  95% CI: [{ppo_ci_lower:.2f}, {ppo_ci_upper:.2f}]\n"
    stats_text += f"  N Seeds: {ppo_data['n_seeds']}\n\n"
    
    stats_text += "Best Baseline:\n"
    best_idx = np.argmax(baseline_means)
    stats_text += f"  Method: {baseline_names[best_idx]}\n"
    stats_text += f"  Mean: {baseline_means[best_idx]:.2f}\n\n"
    
    stats_text += f"PPO Improvement:\n"
    improvement = (ppo_mean - best_baseline_mean) / abs(best_baseline_mean) * 100
    stats_text += f"  Absolute: {ppo_mean - best_baseline_mean:.2f}\n"
    stats_text += f"  Relative: {improvement:.2f}%\n"
    
    axes[1, 1].text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                    fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved baseline comparison plot to: {output_dir / 'baseline_comparison.png'}")
    
    # Create comparison table
    create_comparison_table(ppo_data, baseline_data, output_dir)
    
    return improvement

def create_comparison_table(ppo_data, baseline_data, output_dir):
    """Create comparison table in markdown format."""
    
    baselines = baseline_data['results']
    
    table = "# Performance Comparison Table\n\n"
    table += "| Method | Mean Return | Std Error | 95% CI | vs Baseline | Significance |\n"
    table += "|--------|-------------|-----------|--------|-------------|--------------|\n"
    
    # PPO row
    ppo_mean = ppo_data['mean_return']
    ppo_se = ppo_data['se_return']
    ppo_ci = f"[{ppo_data['ci_lower']:.2f}, {ppo_data['ci_upper']:.2f}]"
    table += f"| **PPO (Ours)** | **{ppo_mean:.2f}** | **{ppo_se:.2f}** | **{ppo_ci}** | - | **-** |\n"
    
    # Baseline rows
    best_baseline = max([b['mean_return'] for b in baselines.values()])
    
    for name, data in baselines.items():
        mean = data['mean_return']
        std = data['std_return']
        ci = f"[{mean:.2f}, {mean:.2f}]"  # No CI for baselines
        
        diff = mean - best_baseline
        rel_diff = diff / abs(best_baseline) * 100 if best_baseline != 0 else 0
        
        table += f"| {name} | {mean:.2f} | {std:.2f} | {ci} | {diff:.2f} ({rel_diff:+.1f}%) | - |\n"
    
    # PPO vs best baseline
    ppo_vs_best = ppo_mean - best_baseline
    ppo_vs_best_rel = ppo_vs_best / abs(best_baseline) * 100 if best_baseline != 0 else 0
    
    table += f"\n**PPO vs Best Baseline:** {ppo_vs_best:.2f} ({ppo_vs_best_rel:+.1f}%)\n"
    
    # Save table
    with open(output_dir / "comparison_table.md", 'w') as f:
        f.write(table)
    
    print(f"Saved comparison table to: {output_dir / 'comparison_table.md'}")

def main():
    """Main function."""
    print("="*60)
    print("CREATING BASELINE COMPARISON PLOTS")
    print("="*60)
    
    # Load data
    ppo_data = load_multi_seed_summary()
    baseline_data = load_baseline_results()
    
    if ppo_data is None:
        print("Error: PPO multi-seed results not found!")
        return
    
    if baseline_data is None:
        print("Error: Baseline results not found!")
        return
    
    print(f"PPO Mean Return: {ppo_data['mean_return']:.2f} ± {ppo_data['se_return']:.2f}")
    print(f"Baselines found: {len(baseline_data['results'])}")
    
    # Create comparison plots
    improvement = create_comparison_plot(ppo_data, baseline_data, "results/final_results")
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON COMPLETE!")
    print("="*60)
    print(f"PPO Improvement: {improvement:.2f}% over best baseline")
    print("="*60)

if __name__ == "__main__":
    main()
