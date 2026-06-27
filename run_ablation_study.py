"""
Ablation Study Runner

Systematically tests impact of each component by running experiments with different configurations:
1. Single objectives (cost only, carbon only, etc.)
2. No safety constraints
3. No offline pre-training (already default)
4. Full multi-objective (baseline)

Usage:
    python run_ablation_study.py --n_seeds 3 --total_timesteps 500000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def create_ablation_config(base_config_path: str, ablation_name: str, output_path: Path) -> Path:
    """
    Create ablation configuration file.
    
    Args:
        base_config_path: Path to base configuration
        ablation_name: Name of ablation
        output_path: Output path for new config
        
    Returns:
        Path to created config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify based on ablation type
    if ablation_name == "cost_only":
        config['reward']['weight_cost'] = 1.0
        config['reward']['weight_carbon'] = 0.0
        config['reward']['weight_peak'] = 0.0
        config['reward']['weight_health'] = 0.0
        
    elif ablation_name == "carbon_only":
        config['reward']['weight_cost'] = 0.0
        config['reward']['weight_carbon'] = 1.0
        config['reward']['weight_peak'] = 0.0
        config['reward']['weight_health'] = 0.0
        
    elif ablation_name == "cost_carbon":
        config['reward']['weight_cost'] = 0.5
        config['reward']['weight_carbon'] = 0.5
        config['reward']['weight_peak'] = 0.0
        config['reward']['weight_health'] = 0.0
        
    elif ablation_name == "no_safety":
        config['environment']['use_safety_constraints'] = False
        
    elif ablation_name == "no_offline":
        config['offline_pretraining']['enabled'] = False
        
    elif ablation_name == "full":
        # Keep default (already correct)
        pass
    
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")
    
    # Save modified config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created ablation config: {output_path}")
    return output_path


def run_ablation_experiment(
    ablation_name: str,
    config_path: Path,
    seed: int,
    total_timesteps: int,
    output_dir: Path
) -> None:
    """
    Run single ablation experiment.
    
    Args:
        ablation_name: Name of ablation
        config_path: Path to config file
        seed: Random seed
        total_timesteps: Total training timesteps
        output_dir: Output directory
    """
    print(f"\nRunning ablation '{ablation_name}' with seed {seed}...")
    
    # Modify config to set seed
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['seed'] = seed
    config['training']['total_timesteps'] = total_timesteps
    
    temp_config = output_dir / "temp_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    cmd = [
        sys.executable,
        "train.py",
        "--config", str(temp_config)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Ablation '{ablation_name}' (seed {seed}) completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"  Error running ablation '{ablation_name}' (seed {seed}): {e}")
        print(f"  Output: {e.output}")


def aggregate_ablation_results(results_dir: Path, ablations: list, n_seeds: int) -> pd.DataFrame:
    """
    Aggregate results from ablation studies.
    
    Args:
        results_dir: Results directory
        ablations: List of ablation names
        n_seeds: Number of seeds per ablation
        
    Returns:
        DataFrame with aggregated results
    """
    data = []
    
    for ablation in ablations:
        ablation_dir = results_dir / ablation
        
        values = []
        for seed in range(n_seeds):
            seed_file = ablation_dir / f"seed_{seed+42}" / "metrics.json"
            
            if seed_file.exists():
                with open(seed_file, 'r') as f:
                    metrics = json.load(f)
                values.append(metrics['mean_return'])
        
        if values:
            data.append({
                'Ablation': ablation,
                'Mean': np.mean(values),
                'SE': np.std(values, ddof=1) / np.sqrt(len(values)),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': len(values)
            })
    
    return pd.DataFrame(data)


def create_ablation_plot(results_df: pd.DataFrame, output_file: Path) -> None:
    """
    Create bar plot for ablation study results.
    
    Args:
        results_df: Results dataframe
        output_file: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by mean performance
    results_df = results_df.sort_values('Mean', ascending=True)
    
    # Create bars
    colors = ['#3498db' if abl != 'full' else '#2ecc71' for abl in results_df['Ablation']]
    bars = ax.barh(results_df['Ablation'], results_df['Mean'], xerr=results_df['SE'], 
                   capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight full model
    full_idx = results_df[results_df['Ablation'] == 'full'].index
    if len(full_idx) > 0:
        bars[list(results_df.index).index(full_idx[0])].set_edgecolor('#27ae60')
        bars[list(results_df.index).index(full_idx[0])].set_linewidth(2)
    
    ax.set_xlabel('Mean Return', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ablation Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study Results (Mean ± SE)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation plot saved to: {output_file}")


def create_ablation_heatmap(results_df: pd.DataFrame, output_file: Path) -> None:
    """
    Create heatmap showing component importance.
    
    Args:
        results_df: Results dataframe
        output_file: Output file path
    """
    # Create matrix for heatmap
    # Rows: Ablations, Columns: Metrics contribution
    
    # For simplicity, we'll show relative performance
    full_performance = results_df[results_df['Ablation'] == 'full']['Mean'].values[0] if 'full' in results_df['Ablation'].values else 0
    
    ablation_map = {
        'cost_only': 'Cost Only',
        'carbon_only': 'Carbon Only',
        'cost_carbon': 'Cost + Carbon',
        'no_safety': 'No Safety',
        'no_offline': 'No Offline',
        'full': 'Full Model'
    }
    
    # Create data for heatmap
    results_df['Label'] = results_df['Ablation'].map(ablation_map)
    results_df['Relative'] = (results_df['Mean'] / full_performance) * 100 if full_performance != 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create simple bar chart as heatmap alternative
    results_df_sorted = results_df.sort_values('Relative', ascending=False)
    
    colors_map = plt.cm.RdYlGn(results_df_sorted['Relative'] / 100)
    bars = ax.barh(results_df_sorted['Label'], results_df_sorted['Relative'], 
                   color=colors_map, edgecolor='black')
    
    ax.set_xlabel('Performance Relative to Full Model (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Importance', fontsize=14, fontweight='bold')
    ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Full Model')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation heatmap saved to: {output_file}")


def main(args):
    """Main ablation study runner."""
    
    print("="*80)
    print("ABLATION STUDY RUNNER")
    print("="*80)
    print(f"Base config: {args.config}")
    print(f"Seeds per ablation: {args.n_seeds}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define ablations to run
    ablations = [
        "cost_only",
        "carbon_only",
        "cost_carbon",
        "no_safety",
        "full"  # Full model as baseline
    ]
    
    print(f"Running {len(ablations)} ablations × {args.n_seeds} seeds = {len(ablations) * args.n_seeds} total runs\n")
    
    # Create configs for each ablation
    configs = {}
    for ablation in ablations:
        config_path = output_dir / f"config_{ablation}.yaml"
        configs[ablation] = create_ablation_config(args.config, ablation, config_path)
    
    # Run experiments
    if not args.skip_training:
        for ablation in ablations:
            print(f"\n{'='*80}")
            print(f"Ablation: {ablation}")
            print(f"{'='*80}")
            
            ablation_dir = output_dir / ablation
            ablation_dir.mkdir(parents=True, exist_ok=True)
            
            for seed_idx in range(args.n_seeds):
                seed = 42 + seed_idx
                run_ablation_experiment(
                    ablation_name=ablation,
                    config_path=configs[ablation],
                    seed=seed,
                    total_timesteps=args.total_timesteps,
                    output_dir=ablation_dir
                )
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING ABLATION RESULTS")
    print("="*80)
    
    results_df = aggregate_ablation_results(output_dir, ablations, args.n_seeds)
    
    print("\nAblation Results:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_file = output_dir / "ablation_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    if len(results_df) > 0:
        print("\nGenerating visualizations...")
        create_ablation_plot(results_df, output_dir / "ablation_comparison.png")
        create_ablation_heatmap(results_df, output_dir / "ablation_importance.png")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Base configuration file"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of seeds per ablation"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500000,
        help="Training timesteps per run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation_study",
        help="Output directory for ablation results"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only aggregate existing results"
    )
    
    args = parser.parse_args()
    main(args)
