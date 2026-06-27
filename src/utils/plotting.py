"""
Visualization and Plotting Utilities

Creates publication-quality plots for:
- Learning curves with confidence intervals
- Baseline comparison bar charts  
- Reward component breakdowns
- Training stability metrics (KL, clip fraction, entropy)
- Ablation study heatmaps
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'


def plot_learning_curves(
    results_files: List[Path],
    labels: List[str],
    output_file: Path,
    title: str = "Learning Curves",
    xlabel: str = "Training Steps",
    ylabel: str = "Episode Return",
    smooth_window: int = 10
):
    """
    Plot learning curves with confidence intervals.
    
    Args:
        results_files: List of paths to results JSON files (one per seed)
        labels: List of labels for each curve
        output_file: Output file path for plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smooth_window: Window size for smoothing
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for label, files in zip(labels, results_files):
       # Load all seeds for this method
        all_returns = []
        all_steps = None
        
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
            
            steps = np.array(data.get('steps', list(range(len(data['returns'])))))
            returns = np.array(data['returns'])
            
            # Smooth if needed
            if smooth_window > 1:
                returns = smooth_reward(returns,window=smooth_window)
            
            all_returns.append(returns)
            if all_steps is None:
                all_steps = steps
        
        # Compute mean and std across seeds
        all_returns = np.array(all_returns)
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        
        # Plot mean with confidence interval
        ax.plot(all_steps, mean_returns, label=label, linewidth=2)
        ax.fill_between(
            all_steps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.2
        )
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Learning curve saved to: {output_file}")
    plt.close()


def plot_baseline_comparison(
    results_file: Path,
    output_file: Path,
    title: str = "Baseline Comparison"
):
    """
    Create bar chart comparing baselines.
    
    Args:
        results_file: Path to baseline comparison JSON
        output_file: Output file path
        title: Plot title
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract method names and means
    methods = list(results.keys())
    means = [results[m]['mean_return'] for m in methods]
    stds = [results[m]['std_return'] for m in methods]
    
    # Sort by performance
    sorted_indices = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    colors = ['#2ecc71' if 'PPO' in m else '#3498db' if 'Rule' in m else '#95a5a6' for m in methods]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel("Method", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Episode Return", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{mean:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Baseline comparison saved to: {output_file}")
    plt.close()


def plot_reward_components(
    log_file: Path,
    output_file: Path,
    title: str = "Reward Component Breakdown"
):
    """
    Plot stacked area chart of reward components over time.
    
    Args:
        log_file: Path to training log with reward components
        output_file: Output file path
        title: Plot title
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    steps = data.get('steps', [])
    cost_comp = data.get('cost_component', [])
    carbon_comp = data.get('carbon_component', [])
    peak_comp = data.get('peak_component', [])
    health_comp = data.get('health_component', [])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(steps, cost_comp, label='Cost', linewidth=2)
    ax.plot(steps, carbon_comp, label='Carbon', linewidth=2)
    ax.plot(steps, peak_comp, label='Peak', linewidth=2)
    ax.plot(steps, health_comp, label='Battery Health', linewidth=2)
    
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Reward Component Value", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Reward components plot saved to: {output_file}")
    plt.close()


def plot_training_stability(
    log_file: Path,
    output_file: Path
):
    """
    Plot training stability metrics (KL div, clip fraction, entropy).
    
    Args:
        log_file: Path to training log
        output_file: Output file path
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    steps = data.get('steps', [])
    kl_div = data.get('kl_divergence', [])
    clip_frac = data.get('clip_fraction', [])
    entropy = data.get('entropy', [])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # KL divergence
    axes[0].plot(steps, kl_div, 'b-', linewidth=2)
    axes[0].axhline(y=0.02, color='r', linestyle='--', label='Target: 0.01-0.02')
    axes[0].set_ylabel("KL Divergence", fontsize=12)
    axes[0].set_title("Training Stability Metrics", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Clip fraction
    axes[1].plot(steps, clip_frac, 'g-', linewidth=2)
    axes[1].axhline(y=0.1, color='r', linestyle='--', label='Target: 5-15%')
    axes[1].axhline(y=0.15, color='r', linestyle='--')
    axes[1].set_ylabel("Clip Fraction", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Entropy
    axes[2].plot(steps, entropy, 'orange', linewidth=2)
    axes[2].set_xlabel("Training Steps", fontsize=12)
    axes[2].set_ylabel("Entropy", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training stability plot saved to: {output_file}")
    plt.close()


def plot_ablation_heatmap(
    results_df: pd.DataFrame,
    output_file: Path,
    title: str = "Ablation Study Results"
):
    """
    Create heatmap for ablation study results.
    
    Args:
        results_df: DataFrame with ablation results
        output_file: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        results_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={'label': 'Mean Return'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Ablation heatmap saved to: {output_file}")
    plt.close()


def smooth_reward(rewards: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to rewards."""
    if window <= 1:
        return rewards
    
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    # Pad to match original length
    pad_left = (len(rewards) - len(smoothed)) // 2
    pad_right = len(rewards) - len(smoothed) - pad_left
    smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
    
    return smoothed


def create_all_plots(results_dir: Path, output_dir: Path):
    """
    Create all standard plots from results directory.
    
    Args:
        results_dir: Directory containing results JSON files
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating publication-quality plots...")
    
    # 1. Learning curves (if training logs exist)
    training_logs = list(results_dir.glob("**/training_log*.json"))
    if training_logs:
        print(f"Found {len(training_logs)} training logs")
        plot_learning_curves(
            [training_logs],
            ["PPO"],
            output_dir / "learning_curves.png",
            title="PPO Training Progress on Smart Grid"
        )
    
    # 2. Baseline comparison (if baseline results exist)
    baseline_files = list(results_dir.glob("**/baseline_comparison*.json"))
    if baseline_files:
        print(f"Found baseline comparison file")
        plot_baseline_comparison(
            baseline_files[0],
            output_dir / "baseline_comparison.png",
            title="PPO vs. Baseline Controllers"
        )
    
    # 3. Reward components (if available)
    reward_logs = list(results_dir.glob("**/reward_components*.json"))
    if reward_logs:
        plot_reward_components(
            reward_logs[0],
            output_dir / "reward_components.png"
        )
    
    # 4. Training stability
    stability_logs = list(results_dir.glob("**/training_log*.json"))
    if stability_logs:
        plot_training_stability(
            stability_logs[0],
            output_dir / "training_stability.png"
        )
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create publication plots")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    create_all_plots(Path(args.results_dir), Path(args.output_dir))
