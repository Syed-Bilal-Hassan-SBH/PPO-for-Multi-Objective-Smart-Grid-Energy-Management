#!/usr/bin/env python3
"""
Analysis and Metrics Script

Analyzes training results, generates plots, and computes statistics for:
- Training curves (returns, losses)
- Baseline comparisons
- Ablation study results
- Statistical significance testing
- Reproducibility analysis across random seeds
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


class ExperimentAnalyzer:
    """Analyzes PPO training experiments."""
    
    def __init__(self, log_dir: str):
        """
        Initialize analyzer.
        
        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir)
        self.results = {}
    
    def load_metrics(self, seed: int = 42) -> Dict:
        """
        Load metrics from a training run.
        
        Args:
            seed: Random seed used in training
            
        Returns:
            Dictionary with metrics
        """
        metrics_path = self.log_dir / 'metrics.json'
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}
    
    def plot_training_curves(self, metrics: Dict, save_path: str = None):
        """
        Plot training curves.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Policy loss
        if 'train/policy_loss' in metrics:
            steps, losses = zip(*metrics['train/policy_loss'])
            axes[0, 0].plot(steps, losses)
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Policy Loss')
            axes[0, 0].set_title('Policy Loss Over Training')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Value loss
        if 'train/value_loss' in metrics:
            steps, losses = zip(*metrics['train/value_loss'])
            axes[0, 1].plot(steps, losses)
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Value Loss')
            axes[0, 1].set_title('Value Loss Over Training')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy
        if 'train/entropy' in metrics:
            steps, entropies = zip(*metrics['train/entropy'])
            axes[1, 0].plot(steps, entropies)
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].set_title('Entropy Over Training')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Returns
        if 'eval/returns' in metrics:
            steps, returns = zip(*metrics['eval/returns'])
            axes[1, 1].plot(steps, returns, 'o-')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Episode Return')
            axes[1, 1].set_title('Evaluation Returns')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Saved training curves to {save_path}")
        
        return fig
    
    def plot_baseline_comparison(self, baselines: Dict[str, Dict], save_path: str = None):
        """
        Plot baseline comparison.
        
        Args:
            baselines: Dictionary {name: metrics}
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        names = list(baselines.keys())
        returns = [baselines[n]['mean_return'] for n in names]
        stds = [baselines[n]['std_return'] for n in names]
        
        # Returns comparison
        colors = ['green' if 'ppo' in n.lower() else 'blue' for n in names]
        axes[0].bar(names, returns, yerr=stds, capsize=5, color=colors, alpha=0.7)
        axes[0].set_ylabel('Mean Return')
        axes[0].set_title('Agent Performance Comparison')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Improvement over best baseline
        best_baseline = max([v for k, v in zip(names, returns) if 'ppo' not in k.lower()])
        improvements = [(r - best_baseline) / abs(best_baseline) * 100 if best_baseline != 0 else 0
                       for r in returns]
        
        axes[1].bar(names, improvements, color=colors, alpha=0.7)
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title(f'Improvement over Best Baseline ({best_baseline:.1f})')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Saved baseline comparison to {save_path}")
        
        return fig
    
    def plot_reward_ablation(self, ablations: Dict[str, Dict], save_path: str = None):
        """
        Plot reward weight ablation.
        
        Args:
            ablations: Dictionary {config_name: metrics}
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        names = list(ablations.keys())
        returns = [ablations[n].get('mean_return', 0) for n in names]
        costs = [ablations[n].get('cost_reduction', 0) for n in names]
        
        # Returns
        axes[0].plot(names, returns, 'o-', linewidth=2, markersize=8)
        axes[0].set_ylabel('Mean Return')
        axes[0].set_title('Reward Weight Ablation: Returns')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cost reduction
        axes[1].plot(names, costs, 's-', linewidth=2, markersize=8, color='orange')
        axes[1].set_ylabel('Cost Reduction (%)')
        axes[1].set_title('Reward Weight Ablation: Cost Reduction')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Saved ablation results to {save_path}")
        
        return fig
    
    def compute_statistics(self, results_list: List[Dict]) -> Dict:
        """
        Compute statistical measures across multiple runs.
        
        Args:
            results_list: List of result dictionaries from different seeds
            
        Returns:
            Dictionary with statistics
        """
        returns = [r.get('mean_return', 0) for r in results_list]
        
        stats_dict = {
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
            'min': float(np.min(returns)),
            'max': float(np.max(returns)),
            'ci_95': float(1.96 * np.std(returns) / np.sqrt(len(returns))),
            'n_runs': len(results_list),
            'values': returns
        }
        
        return stats_dict
    
    def test_significance(self, group1: List[float], group2: List[float]) -> Dict:
        """
        Test statistical significance between two groups.
        
        Args:
            group1: Returns from group 1
            group2: Returns from group 2
            
        Returns:
            T-test results
        """
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'mean_diff': float(np.mean(group1) - np.mean(group2)),
            'cohens_d': float((np.mean(group1) - np.mean(group2)) / 
                              np.sqrt((np.var(group1) + np.var(group2)) / 2))
        }
    
    def generate_report(self, output_dir: str = 'analysis_report'):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        report = {
            'timestamp': str(Path().resolve()),
            'sections': {}
        }
        
        print("=" * 60)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 60)
        
        # Load metrics
        metrics = self.load_metrics()
        if metrics:
            # Plot training curves
            self.plot_training_curves(metrics, 
                                     save_path=str(output_path / 'training_curves.png'))
            report['sections']['training_curves'] = {
                'generated': True,
                'path': 'training_curves.png'
            }
        
        # Save report
        report_path = output_path / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {report_path}")


def analyze_results():
    """Main analysis function."""
    analyzer = ExperimentAnalyzer('results/logs')
    
    # Generate report
    analyzer.generate_report('results/analysis')
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    analyze_results()
