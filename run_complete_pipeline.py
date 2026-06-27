"""
Full Publication Pipeline

One-click script to run complete publication-ready experimental suite:
1. Multi-seed PPO training (5 seeds)
2. Baseline evaluation (5 baselines)
3. Ablation studies (5 configurations × 3 seeds)
4. Statistical analysis
5. Publication-quality visualizations
6. Result aggregation and reporting

Usage:
    # Full pipeline (takes 15-20 hours on CPU, 3-5 hours on GPU)
    python run_full_publication_pipeline.py --full
    
    # Quick test (100K steps, 3 seeds)
    python run_full_publication_pipeline.py --quick_test
    
    # Skip training and only analyze existing results
    python run_full_publication_pipeline.py --skip_training
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import json


def run_command(cmd: list, description: str) -> bool:
    """
    Run command and handle errors.
    
    Args:
        cmd: Command to run
        description: Description of step
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False


def estimate_time(total_timesteps: int, n_seeds: int, n_ablations: int, use_gpu: bool = False) -> str:
    """
    Estimate total pipeline duration.
    
    Args:
        total_timesteps: Timesteps per run
        n_seeds: Number of seeds for main experiment
        n_ablations: Number of ablation runs
        use_gpu: Whether GPU is available
        
    Returns:
        Estimate string
    """
    # Time per 1M steps
    time_per_1m = 2 if use_gpu else 12  # hours
    
    # Main experiments
    main_time = (total_timesteps / 1_000_000) * time_per_1m * n_seeds
    
    # Ablations (typically run with fewer steps)
    ablation_time = (total_timesteps * 0.5 / 1_000_000) * time_per_1m * n_ablations
    
    # Baseline evaluation
    baseline_time = 0.5  # hours
    
    # Analysis
    analysis_time = 0.5  # hours
    
    total_hours = main_time + ablation_time + baseline_time + analysis_time
    
    return f"{total_hours:.1f} hours ({total_hours/24:.1f} days)"


def main(args):
    """Main pipeline orchestrator."""
    
    print()
    print("="*80)
    print(" "*25 + "PUBLICATION PIPELINE")
    print("="*80)
    print()
    
    start_time = datetime.now()
    
    # Determine settings based on mode
    if args.quick_test:
        total_timesteps = 100000
        n_seeds = 3
        n_ablation_seeds = 2
        mode = "QUICK TEST"
    elif args.full:
        total_timesteps = 1000000
        n_seeds = 5
        n_ablation_seeds = 3
        mode = "FULL PUBLICATION"
    else:
        total_timesteps = args.total_timesteps
        n_seeds = args.n_seeds
        n_ablation_seeds = args.ablation_seeds
        mode = "CUSTOM"
    
    print(f"Mode: {mode}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Main experiment seeds: {n_seeds}")
    print(f"Ablation seeds: {n_ablation_seeds}")
    print(f"Skip training: {args.skip_training}")
    print()
    
    # Estimate time
    if not args.skip_training:
        estimate = estimate_time(total_timesteps, n_seeds, 5 * n_ablation_seeds, use_gpu=False)
        print(f"⏱️  Estimated duration: {estimate}")
        print()
        
        if not args.quick_test and not args.full:
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    results_summary = {
        'start_time': start_time.isoformat(),
        'mode': mode,
        'total_timesteps': total_timesteps,
        'n_seeds': n_seeds,
        'steps_completed': []
    }
    
    # Phase 1: Multi-Seed Training
    if not args.skip_training:
        success = run_command(
            [
                sys.executable, "run_multi_seed.py",
                "--n_seeds", str(n_seeds),
                "--total_timesteps", str(total_timesteps),
                "--config", args.config,
                "--output_dir", "results/multi_seed"
            ],
            f"Phase 1: Multi-Seed Training ({n_seeds} seeds × {total_timesteps:,} steps)"
        )
        
        results_summary['steps_completed'].append({
            'phase': 'Multi-Seed Training',
            'success': success
        })
        
        if not success and not args.continue_on_error:
            print("\n❌ Pipeline failed at Phase 1")
            return
    else:
        print("\n⏭️  Skipping Phase 1: Multi-Seed Training")
        results_summary['steps_completed'].append({
            'phase': 'Multi-Seed Training',
            'success': 'skipped'
        })
    
    # Phase 2: Baseline Evaluation
    if not args.skip_baselines:
        success = run_command(
            [
                sys.executable, "evaluate_baselines.py",
                "--n_episodes", "20",
                "--output_dir", "results/multi_seed/baselines"
            ],
            "Phase 2: Baseline Evaluation"
        )
        
        results_summary['steps_completed'].append({
            'phase': 'Baseline Evaluation',
            'success': success
        })
        
        if not success and not args.continue_on_error:
            print("\n❌ Pipeline failed at Phase 2")
            return
    else:
        print("\n⏭️  Skipping Phase 2: Baseline Evaluation")
        results_summary['steps_completed'].append({
            'phase': 'Baseline Evaluation',
            'success': 'skipped'
        })
    
    # Phase 3: Ablation Studies
    if not args.skip_ablations and not args.skip_training:
        ablation_timesteps = total_timesteps // 2  # Half timesteps for ablations
        
        success = run_command(
            [
                sys.executable, "run_ablation_study.py",
                "--n_seeds", str(n_ablation_seeds),
                "--total_timesteps", str(ablation_timesteps),
                "--config", args.config,
                "--output_dir", "results/ablation_study"
            ],
            f"Phase 3: Ablation Studies ({n_ablation_seeds} seeds × {ablation_timesteps:,} steps)"
        )
        
        results_summary['steps_completed'].append({
            'phase': 'Ablation Studies',
            'success': success
        })
        
        if not success and not args.continue_on_error:
            print("\n❌ Pipeline failed at Phase 3")
            return
    else:
        print("\n⏭️  Skipping Phase 3: Ablation Studies")
        results_summary['steps_completed'].append({
            'phase': 'Ablation Studies',
            'success': 'skipped'
        })
    
    # Phase 4: Result Aggregation
    success = run_command(
        [
            sys.executable, "aggregate_results.py",
            "--results_dir", "results/multi_seed",
            "--n_seeds", str(n_seeds),
            "--output_dir", "results/final_results"
        ],
        "Phase 4: Result Aggregation & Statistical Analysis"
    )
    
    results_summary['steps_completed'].append({
        'phase': 'Result Aggregation',
        'success': success
    })
    
    if not success and not args.continue_on_error:
        print("\n❌ Pipeline failed at Phase 4")
        return
    
    # Phase 5: Visualization Generation
    success = run_command(
        [
            sys.executable, "generate_results.py",
            "--results_dir", "results/multi_seed",
            "--output_dir", "results/final_results/plots"
        ],
        "Phase 5: Publication-Quality Visualizations"
    )
    
    results_summary['steps_completed'].append({
        'phase': 'Visualization Generation',
        'success': success
    })
    
    # Final Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    results_summary['end_time'] = end_time.isoformat()
    results_summary['total_duration_seconds'] = duration.total_seconds()
    results_summary['total_duration_hours'] = duration.total_seconds() / 3600
    
    # Save summary
    summary_file = Path("results/final_results/pipeline_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print()
    print("="*80)
    print(" "*22 + "PIPELINE COMPLETE! 🎉")
    print("="*80)
    print()
    print(f"Total duration: {duration.total_seconds()/3600:.2f} hours")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("✅ Generated outputs:")
    print("  📊 Statistical reports: results/final_results/")
    print("  📈 Publication plots: results/final_results/plots/")
    print("  📋 Comparison tables: results/final_results/comparison_table.*")
    print("  📄 Ablation results: results/ablation_study/")
    print()
    print("📝 Next steps:")
    print("  1. Review results in results/final_results/")
    print("  2. Check statistical significance in statistical_report.md")
    print("  3. Use plots in your paper/presentation")
    print("  4. Copy comparison_table.tex to your LaTeX paper")
    print()
    print(f"Pipeline summary saved to: {summary_file}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full publication pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2-3 hours)
  python run_full_publication_pipeline.py --quick_test
  
  # Full publication run (15-20 hours on CPU)
  python run_full_publication_pipeline.py --full
  
  # Custom configuration
  python run_full_publication_pipeline.py --n_seeds 5 --total_timesteps 1000000
  
  # Only analyze existing results
  python run_full_publication_pipeline.py --skip_training
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode (100K steps, 3 seeds)"
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full publication mode (1M steps, 5 seeds)"
    )
    
    # Custom parameters
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Base configuration file"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds for main experiment"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Training timesteps per seed"
    )
    parser.add_argument(
        "--ablation_seeds",
        type=int,
        default=3,
        help="Number of seeds for ablation studies"
    )
    
    # Skip options
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip all training and only run analysis"
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--skip_ablations",
        action="store_true",
        help="Skip ablation studies"
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.quick_test and not args.full:
        if args.n_seeds < 3:
            parser.error("Minimum 3 seeds required for statistical significance")
    
    main(args)
