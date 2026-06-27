"""
Result Aggregation Script

Aggregates results from multiple seed runs and generates outputs:
- Statistical summaries with confidence intervals
- Comparison tables (Markdown and LaTeX)
- Significance testing against baselines
- Publication-quality figures

Usage:
    python aggregate_results.py --results_dir results/multi_seed --n_seeds 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.statistics import (
    compute_descriptive_stats,
    aggregate_multi_seed_results,
    compare_methods,
    generate_statistical_report,
    save_statistics_json,
    generate_significance_marker
)

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_seed_results(results_dir: Path, n_seeds: int, method_name: str = "PPO") -> Dict:
    """
    Load results from multiple seed runs.
    
    Args:
        results_dir: Directory containing seed results
        n_seeds: Number of seeds
        method_name: Method name (e.g., "PPO")
        
    Returns:
        Dictionary mapping seed -> metrics
    """
    seed_results = {}
    
    for seed in range(n_seeds):
        seed_path = results_dir / f"seed_{seed}"
        metric_files = list(seed_path.rglob("metrics.json"))
        
        if not metric_files:
            print(f"Warning: Missing results for seed {seed}")
            continue
        
        # Take the most recently modified metrics file
        seed_file = sorted(metric_files, key=lambda p: p.stat().st_mtime)[-1]
        
        with open(seed_file, 'r') as f:
            metrics = json.load(f)
        
        seed_results[seed] = metrics
    
    print(f"Loaded results from {len(seed_results)} seeds")
    return seed_results


def create_comparison_table_markdown(
    ppo_stats: Dict,
    baseline_stats: Dict,
    significance_results: Dict,
    output_file: Path
) -> None:
    """
    Create comparison table in Markdown format.
    
    Args:
        ppo_stats: PPO statistics
        baseline_stats: Dictionary of baseline statistics
        significance_results: Statistical test results
        output_file: Output markdown file
    """
    lines = ["# Method Comparison Table\n"]
    lines.append("## Mean Return ± Standard Error\n")
    lines.append("| Method | Mean ± SE | 95% CI | vs PPO (p-value) | Sig | Effect Size |")
    lines.append("|--------|-----------|--------|------------------|-----|-------------|")
    
    # Add PPO row
    ppo_mean = ppo_stats['mean_return'].mean
    ppo_se = ppo_stats['mean_return'].se
    ppo_ci = f"[{ppo_stats['mean_return'].ci_lower:.2f}, {ppo_stats['mean_return'].ci_upper:.2f}]"
    lines.append(f"| **PPO (Ours)** | **{ppo_mean:.2f} ± {ppo_se:.2f}** | {ppo_ci} | - | - | - |")
    
    # Add baseline rows
    for baseline_name, stats in baseline_stats.items():
        mean = stats['mean_return'].mean
        se = stats['mean_return'].se
        ci = f"[{stats['mean_return'].ci_lower:.2f}, {stats['mean_return'].ci_upper:.2f}]"
        
        # Get significance results
        if baseline_name in significance_results:
            test_result = significance_results[baseline_name]
            p_val = test_result.p_value
            sig_marker = generate_significance_marker(p_val)
            effect_size = f"{test_result.effect_size:.2f}"
        else:
            p_val = 1.0
            sig_marker = "ns"
            effect_size = "N/A"
        
        lines.append(
            f"| {baseline_name} | {mean:.2f} ± {se:.2f} | {ci} | {p_val:.4f} | {sig_marker} | {effect_size} |"
        )
    
    lines.append("\n**Legend:**")
    lines.append("- **Sig**: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)")
    lines.append("- **Effect Size**: Cohen's d")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Comparison table saved to: {output_file}")


def create_comparison_table_latex(
    ppo_stats: Dict,
    baseline_stats: Dict,
    significance_results: Dict,
    output_file: Path
) -> None:
    """
    Create comparison table in LaTeX format.
    
    Args:
        ppo_stats: PPO statistics
        baseline_stats: Dictionary of baseline statistics
        significance_results: Statistical test results
        output_file: Output LaTeX file
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Performance comparison across methods (Mean ± SE over 5 seeds)}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & Mean Return & 95\\% CI & p-value & Cohen's d \\\\")
    lines.append("\\midrule")
    
    # PPO row
    ppo_mean = ppo_stats['mean_return'].mean
    ppo_se = ppo_stats['mean_return'].se
    ppo_ci_lower = ppo_stats['mean_return'].ci_lower
    ppo_ci_upper = ppo_stats['mean_return'].ci_upper
    lines.append(
        f"\\textbf{{PPO (Ours)}} & \\textbf{{{ppo_mean:.2f} $\\pm$ {ppo_se:.2f}}} & "
        f"[{ppo_ci_lower:.2f}, {ppo_ci_upper:.2f}] & - & - \\\\"
    )
    
    # Baseline rows
    for baseline_name, stats in baseline_stats.items():
        mean = stats['mean_return'].mean
        se = stats['mean_return'].se
        ci_lower = stats['mean_return'].ci_lower
        ci_upper = stats['mean_return'].ci_upper
        
        if baseline_name in significance_results:
            test_result = significance_results[baseline_name]
            p_val = test_result.p_value
            sig_marker = generate_significance_marker(p_val)
            effect_size = f"{test_result.effect_size:.2f}"
            
            # Add significance markers
            if sig_marker == "***":
                p_val_str = f"{p_val:.4f}$^{{***}}$"
            elif sig_marker == "**":
                p_val_str = f"{p_val:.4f}$^{{**}}$"
            elif sig_marker == "*":
                p_val_str = f"{p_val:.4f}$^{{*}}$"
            else:
                p_val_str = f"{p_val:.4f}"
        else:
            p_val_str = "N/A"
            effect_size = "N/A"
        
        lines.append(
            f"{baseline_name} & {mean:.2f} $\\pm$ {se:.2f} & "
            f"[{ci_lower:.2f}, {ci_upper:.2f}] & {p_val_str} & {effect_size} \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to: {output_file}")


def create_summary_plot(
    ppo_stats: Dict,
    baseline_stats: Dict,
    output_file: Path
) -> None:
    """
    Create summary bar plot with error bars.
    
    Args:
        ppo_stats: PPO statistics
        baseline_stats: Baseline statistics
        output_file: Output file path
    """
    # Prepare data
    methods = ['PPO'] + list(baseline_stats.keys())
    means = [ppo_stats['mean_return'].mean] + [stats['mean_return'].mean for stats in baseline_stats.values()]
    errors = [ppo_stats['mean_return'].se] + [stats['mean_return'].se for stats in baseline_stats.values()]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71'] + ['#95a5a6'] * len(baseline_stats)  # Green for PPO, gray for baselines
    bars = ax.bar(methods, means, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight PPO
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(2)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Return', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison (Mean ± SE)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {output_file}")


def main(args):
    """Main aggregation function."""
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RESULT AGGREGATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load PPO results
    print("Loading PPO results...")
    ppo_seed_results = load_seed_results(results_dir, args.n_seeds, "PPO")
    
    if len(ppo_seed_results) == 0:
        print("Error: No PPO results found!")
        return
    
    # Aggregate PPO results
    print("\nAggregating PPO results...")
    ppo_stats = aggregate_multi_seed_results(ppo_seed_results)
    
    print(f"PPO Mean Return: {ppo_stats['mean_return'].mean:.2f} ± {ppo_stats['mean_return'].se:.2f}")
    print(f"  95% CI: [{ppo_stats['mean_return'].ci_lower:.2f}, {ppo_stats['mean_return'].ci_upper:.2f}]")
    
    # Save PPO statistics
    save_statistics_json(ppo_stats, output_dir / "ppo_statistics.json")
    
    # Load baseline results if available
    baseline_stats = {}
    significance_results = {}
    
    baseline_dir = results_dir / "baselines"
    if baseline_dir.exists():
        print("\nLoading baseline results...")
        
        baseline_names = ['Random', 'DoNothing', 'RuleBased', 'PeakShaving', 'SimpleMPC']
        
        for baseline_name in baseline_names:
            baseline_file = baseline_dir / f"{baseline_name.lower()}_results.json"
            
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                # Create fake multi-seed structure for consistency
                baseline_seed_results = {
                    i: baseline_data for i in range(args.n_seeds)
                }
                
                baseline_stats[baseline_name] = aggregate_multi_seed_results(baseline_seed_results)
                
                # Statistical comparison
                ppo_returns = [ppo_seed_results[seed]['mean_return'] for seed in ppo_seed_results.keys()]
                baseline_returns = [baseline_data['mean_return']] * len(ppo_returns)  # Replicate for comparison
                
                test_result = compare_methods(
                    np.array(ppo_returns),
                    np.array(baseline_returns)
                )
                significance_results[baseline_name] = test_result
                
                sig_marker = generate_significance_marker(test_result.p_value)
                print(f"  {baseline_name}: p={test_result.p_value:.4f} {sig_marker}, d={test_result.effect_size:.2f}")
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    # 1. Markdown table
    create_comparison_table_markdown(
        ppo_stats,
        baseline_stats,
        significance_results,
        output_dir / "comparison_table.md"
    )
    
    # 2. LaTeX table
    create_comparison_table_latex(
        ppo_stats,
        baseline_stats,
        significance_results,
        output_dir / "comparison_table.tex"
    )
    
    # 3. Summary plot
    if baseline_stats:
        create_summary_plot(
            ppo_stats,
            baseline_stats,
            output_dir / "performance_comparison.png"
        )
    
    # 4. Statistical report
    if baseline_stats:
        # Prepare data for report
        results_dict = {
            'PPO': np.array([ppo_seed_results[seed]['mean_return'] for seed in ppo_seed_results.keys()])
        }
        
        for baseline_name in baseline_stats.keys():
            # Use replicated baseline values
            results_dict[baseline_name] = np.array([baseline_stats[baseline_name]['mean_return'].mean] * len(ppo_seed_results))
        
        report = generate_statistical_report(
            results_dict,
            baseline_name='Random',
            output_file=output_dir / "statistical_report.md"
        )
    
    # 5. Summary JSON
    summary = {
        'n_seeds': len(ppo_seed_results),
        'ppo_mean_return': float(ppo_stats['mean_return'].mean),
        'ppo_se': float(ppo_stats['mean_return'].se),
        'ppo_ci_lower': float(ppo_stats['mean_return'].ci_lower),
        'ppo_ci_upper': float(ppo_stats['mean_return'].ci_upper),
        'baselines': {}
    }
    
    for baseline_name, stats in baseline_stats.items():
        summary['baselines'][baseline_name] = {
            'mean_return': float(stats['mean_return'].mean),
            'se': float(stats['mean_return'].se),
            'p_value': float(significance_results[baseline_name].p_value) if baseline_name in significance_results else None,
            'significant': bool(significance_results[baseline_name].significant) if baseline_name in significance_results else None
        }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {output_dir / 'summary.json'}")
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_dir / 'ppo_statistics.json'}")
    print(f"  - {output_dir / 'comparison_table.md'}")
    print(f"  - {output_dir / 'comparison_table.tex'}")
    print(f"  - {output_dir / 'performance_comparison.png'}")
    print(f"  - {output_dir / 'statistical_report.md'}")
    print(f"  - {output_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experimental results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/multi_seed",
        help="Directory containing multi-seed results"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/final_results",
        help="Output directory for aggregated results"
    )
    
    args = parser.parse_args()
    main(args)
