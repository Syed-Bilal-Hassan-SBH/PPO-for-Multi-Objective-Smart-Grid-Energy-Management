"""
Multi-Seed Experiment Runner

Runs complete PPO training with multiple seeds for statistical significance.
Automatically evaluates against baselines and generates all plots.

Usage:
    python run_multi_seed.py --n_seeds 5 --total_timesteps 1000000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import numpy as np

def run_experiment(seed: int, config: str, total_timesteps: int, output_dir: Path):
    """
    Run single training experiment with given seed.
    
    Args:
        seed: Random seed
        config: Config file path
        total_timesteps: Total training timesteps
        output_dir: Output directory for this seed
    """
    import yaml
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT - SEED {seed}")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Modify config for this seed
    config_data['training']['seed'] = seed
    config_data['training']['total_timesteps'] = total_timesteps
    config_data['logging']['log_dir'] = str(output_dir / "logs")
    config_data['logging']['checkpoint_dir'] = str(output_dir / "checkpoints")
    
    # Save temporary config
    temp_config = output_dir / f"config_seed_{seed}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # Run training with temporary config
    cmd = [
        sys.executable,
        "train.py",
        "--config", str(temp_config)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for seed {seed}")
        return False
    
    print(f"\n✓ Training completed successfully for seed {seed}")
    return True




def run_baseline_evaluation(seed: int, checkpoint_path: Path, output_dir: Path):
    """
    Evaluate against baselines.
    
    Args:
        seed: Random seed
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING BASELINES - SEED {seed}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "evaluate_baselines.py",
        "--ppo_checkpoint", str(checkpoint_path),
        "--seed", str(seed),
        "--n_episodes", "10",
        "--output_dir", str(output_dir)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Baseline evaluation failed for seed {seed}")
        return False
    
    print(f"\n✓ Baseline evaluation completed for seed {seed}")
    return True


def aggregate_results(results_dir: Path, n_seeds: int):
    """
    Aggregate results across all seeds.
    
    Args:
        results_dir: Directory containing all seed results
        n_seeds: Number of seeds
    """
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*80}\n")
    
    all_returns = []
    all_baseline_results = {}
    
    for seed in range(n_seeds):
        seed_dir = results_dir / f"seed_{seed}"
        
        # Load baseline results
        baseline_file = seed_dir / "baselines" / f"baseline_comparison_seed{seed}.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                for method, result in data['results'].items():
                    if method not in all_baseline_results:
                        all_baseline_results[method] = {'returns': []}
                    all_baseline_results[method]['returns'].extend(result['returns'])
    
    # Compute aggregated statistics
    aggregated = {}
    for method, data in all_baseline_results.items():
        returns = data['returns']
        aggregated[method] = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'stderr_return': float(np.std(returns) / np.sqrt(len(returns))),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'n_episodes': len(returns)
        }
    
    # Save aggregated results
    output_file = results_dir / "aggregated_results.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary table
    print("\nAGGREGATED RESULTS (ALL SEEDS):\n")
    print(f"{'Method':<20} | {'Mean ± SE':<20} | {'Min':<10} | {'Max':<10} | {'N':<6}")
    print("-" * 75)
    
    for method in sorted(aggregated.keys(), key=lambda x: aggregated[x]['mean_return'], reverse=True):
        stats = aggregated[method]
        print(f"{method:<20} | {stats['mean_return']:>7.2f} ± {stats['stderr_return']:>6.2f}     | "
              f"{stats['min_return']:>8.2f}   | {stats['max_return']:>8.2f}   | {stats['n_episodes']:<6}")
    
    print(f"\n✓ Aggregated results saved to: {output_file}")


def generate_all_plots(results_dir: Path):
    """
    Generate all publication plots.
    
    Args:
        results_dir: Results directory
    """
    print(f"\n{'='*80}")
    print("GENERATING PUBLICATION PLOTS")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "-m", "src.utils.plotting",
        "--results_dir", str(results_dir),
        "--output_dir", str(results_dir / "plots")
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("WARNING: Plot generation encountered errors")
        return False
    
    print(f"\n✓ All plots generated successfully")
    return True


def main(args):
    """Main experiment runner."""
    
    print(f"\n{'='*80}")
    print("MULTI-SEED PPO SMART GRID EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Seeds: {args.n_seeds}")
    print(f"  - Total timesteps per seed: {args.total_timesteps:,}")
    print(f"  - Config file: {args.config}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    successful_seeds = []
    
    # Run experiments for each seed
    for seed in range(args.n_seeds):
        seed_dir = results_dir / f"seed_{seed}"
        
        # Train
        success = run_experiment(
            seed=seed,
            config=args.config,
            total_timesteps=args.total_timesteps,
            output_dir=seed_dir
        )
        
        if not success:
            print(f"Skipping seed {seed} due to training failure")
            continue
        
        # Find latest checkpoint
        checkpoint_dir = seed_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("model_*.pt"))
        if not checkpoints:
            print(f"No checkpoints found for seed {seed}, skipping evaluation")
            continue
        
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
        
        # Evaluate against baselines
        baseline_success = run_baseline_evaluation(
            seed=seed,
            checkpoint_path=latest_checkpoint,
            output_dir=seed_dir / "baselines"
        )
        
        if baseline_success:
            successful_seeds.append(seed)
    
    # Aggregate results
    if len(successful_seeds) > 0:
        print(f"\n✓ Successfully completed {len(successful_seeds)}/{args.n_seeds} experiments")
        aggregate_results(results_dir, len(successful_seeds))
        
        # Generate plots
        if args.generate_plots:
            generate_all_plots(results_dir)
    else:
        print(f"\n✗ No experiments completed successfully")
        return
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    print(f"Successful seeds: {successful_seeds}")
    print(f"\nNext steps:")
    print(f"  1. Review aggregated results: {results_dir}/aggregated_results.json")
    print(f"  2. Check plots: {results_dir}/plots/")
    print(f"  3. Use results for paper writing")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed PPO experiments")
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds to run"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps per seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/multi_seed",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        default=True,
        help="Generate plots after experiments"
    )
    
    args = parser.parse_args()
    main(args)
