"""
Comprehensive Baseline Evaluation Script

Evaluates PPO agent against all baselines:
- Random
- Do-Nothing
- Rule-Based TOU
- Peak Shaving
- Simple MPC

Performs statistical significance testing and generates comparison tables.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.ppo import PPOAgent
from src.agents.baselines import (
    RandomBaseline,
    DoNothingBaseline,
    RuleBasedBaseline,
    PeakShavingBaseline,
    SimpleModelPredictiveBaseline,
    BaselineEvaluator
)
from src.envs.citylearn_wrapper import CityLearnWrapper
from src.utils.config import load_config
from src.utils.seeding import set_seed
from scipy import stats


def create_dummy_env(n_buildings: int = 5, obs_dim_per_building: int = 15, action_dim_per_building: int = 1):
    """Create dummy environment for testing if CityLearn not available.
    
    Creates an environment that mimics CityLearn's multi-building structure.
    """
    import gymnasium as gym
    from gymnasium import spaces
    
    class DummyMultiBuildingEnv(gym.Env):
        def __init__(self, n_buildings, obs_dim_per_building, action_dim_per_building):
            self.n_buildings = n_buildings
            self.obs_dim_per_building = obs_dim_per_building
            self.action_dim_per_building = action_dim_per_building
            
            # CityLearn uses list of spaces (one per building)
            self.observation_space = [
                spaces.Box(low=-10, high=10, shape=(obs_dim_per_building,), dtype='float32')
                for _ in range(n_buildings)
            ]
            self.action_space = [
                spaces.Box(low=-1, high=1, shape=(action_dim_per_building,), dtype='float32')
                for _ in range(n_buildings)
            ]
            
            self.step_count = 0
            self.max_steps = 8760  # One year hourly
        
        def reset(self):
            self.step_count = 0
            # Return list of observations (one per building)
            obs = [space.sample().astype('float32') for space in self.observation_space]
            return obs, {}
        
        def step(self, actions):
            """
            Args:
                actions: List of actions (one per building)
            """
            self.step_count += 1
            # Return list of observations (one per building)
            obs = [space.sample().astype('float32') for space in self.observation_space]
            # CityLearn returns list of rewards (one per building)
            rewards = [float(np.random.randn() - 0.5) for _ in range(self.n_buildings)]
            done = self.step_count >= self.max_steps
            info = {
                'cost': np.random.uniform(0, 100),
                'carbon': np.random.uniform(0, 10),
                'violations': 0
            }
            return obs, rewards, done, False, info
        
        def close(self):
            pass
    
    return DummyMultiBuildingEnv(n_buildings, obs_dim_per_building, action_dim_per_building)


def evaluate_all_baselines(
    env,
    ppo_agent=None,
    n_episodes: int = 10,
    seed: int = 42
) -> Dict:
    """
    Evaluate all baselines and PPO agent.
    
    Args:
        env: Environment
        ppo_agent: Trained PPO agent (optional)
        n_episodes: Number of episodes per baseline
        seed: Random seed
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.obs_dim
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_dim
    
    # Create baselines
    baselines = {
        'Random': RandomBaseline(action_dim, seed=seed),
        'Do-Nothing': DoNothingBaseline(action_dim),
        'Rule-Based TOU': RuleBasedBaseline(action_dim, n_buildings=5, seed=seed),
        'Peak Shaving': PeakShavingBaseline(action_dim, n_buildings=5, seed=seed),
        'Simple MPC': SimpleModelPredictiveBaseline(action_dim, n_buildings=5, seed=seed)
    }
    
    print("=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    print(f"Episodes per baseline: {n_episodes}")
    print(f"Seed: {seed}")
    print()
    
    # Evaluate each baseline
    for name, baseline in baselines.items():
        print(f"\nEvaluating {name}...")
        result = BaselineEvaluator.evaluate_baseline(baseline, env, n_episodes)
        results[name] = result
        
        print(f"  Mean Return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
        print(f"  Range: [{result['min_return']:.2f}, {result['max_return']:.2f}]")
    
    # Evaluate PPO if provided
    if ppo_agent is not None:
        print(f"\nEvaluating PPO Agent...")
        ppo_returns = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_return = 0.0
            
            while not done:
                action, _, _ = ppo_agent.get_action(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_return += reward
                done = done or truncated
            
            ppo_returns.append(ep_return)
        
        results['PPO'] = {
            'returns': ppo_returns,
            'mean_return': float(np.mean(ppo_returns)),
            'std_return': float(np.std(ppo_returns)),
            'max_return': float(np.max(ppo_returns)),
            'min_return': float(np.min(ppo_returns))
        }
        
        print(f"  Mean Return: {results['PPO']['mean_return']:.2f} ± {results['PPO']['std_return']:.2f}")
        print(f"  Range: [{results['PPO']['min_return']:.2f}, {results['PPO']['max_return']:.2f}]")
    
    return results


def compute_statistical_significance(results: Dict) -> Dict:
    """
    Compute statistical significance between methods.
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        Dictionary with p-values between all pairs
    """
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)
    
    methods = list(results.keys())
    p_values = {}
    
    # Compare each pair
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            returns1 = results[method1]['returns']
            returns2 = results[method2]['returns']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(returns1, returns2)
            
            # Also compute Wilcoxon rank-sum (non-parametric)
            u_stat, p_value_wilcoxon = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
            
            pair_key = f"{method1} vs {method2}"
            p_values[pair_key] = {
                't_test_p_value': float(p_value),
                'wilcoxon_p_value': float(p_value_wilcoxon),
                'mean_diff': results[method1]['mean_return'] - results[method2]['mean_return']
            }
            
            # Print results
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"\n{pair_key}:")
            print(f"  Mean difference: {p_values[pair_key]['mean_diff']:.2f}")
            print(f"  t-test p-value: {p_value:.4f} {significance}")
            print(f"  Wilcoxon p-value: {p_value_wilcoxon:.4f}")
    
    return p_values


def generate_comparison_table(results: Dict) -> str:
    """
    Generate formatted comparison table.
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        Formatted table string
    """
    table = "\n" + "=" * 80 + "\n"
    table += "PERFORMANCE COMPARISON TABLE\n"
    table += "=" * 80 + "\n\n"
    
    table += f"{'Method':<20} | {'Mean Return':<15} | {'Std Return':<12} | {'Min':<10} | {'Max':<10}\n"
    table += "-" * 80 + "\n"
    
    # Sort by mean return (descending)
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['mean_return'], reverse=True)
    
    for method, result in sorted_methods:
        table += f"{method:<20} | "
        table += f"{result['mean_return']:>12.2f}    | "
        table += f"{result['std_return']:>10.2f}   | "
        table += f"{result['min_return']:>8.2f}   | "
        table += f"{result['max_return']:>8.2f}\n"
    
    return table


def save_results(results: Dict, p_values: Dict, output_file: Path):
    """Save evaluation results to JSON."""
    output = {
        'results': results,
        'statistical_tests': p_values,
        'metadata': {
            'n_baselines': len(results),
            'best_method': max(results.items(), key=lambda x: x[1]['mean_return'])[0]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main(args):
    """Main evaluation function."""
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create environment
    try:
        from citylearn import CityLearnEnv
        print("Loading CityLearn environment...")
        base_env = CityLearnEnv(schema="citylearn_challenge_2022_phase_1")
        env = CityLearnWrapper(base_env)
        print("CityLearn environment loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load CityLearn ({e}). Using dummy environment.")
        base_env = create_dummy_env(n_buildings=5)
        env = CityLearnWrapper(base_env)
    
    # Load PPO agent if checkpoint provided
    ppo_agent = None
    if args.ppo_checkpoint:
        print(f"\nLoading PPO agent from: {args.ppo_checkpoint}")
        ppo_agent = PPOAgent(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim
        )
        ppo_agent.load(args.ppo_checkpoint)
        print("PPO agent loaded successfully!")
    
    # Evaluate all baselines
    results = evaluate_all_baselines(
        env,
        ppo_agent=ppo_agent,
        n_episodes=args.n_episodes,
        seed=args.seed
    )
    
    # Compute statistical significance
    p_values = compute_statistical_significance(results)
    
    # Generate and print comparison table
    table = generate_comparison_table(results)
    print(table)
    
    # Save results
    output_file = output_dir / f"baseline_comparison_seed{args.seed}.json"
    save_results(results, p_values, output_file)
    
    # Save table to text file
    table_file = output_dir / f"baseline_table_seed{args.seed}.txt"
    with open(table_file, 'w') as f:
        f.write(table)
    print(f"Table saved to: {table_file}")
    
    env.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baselines for PPO Smart Grid project")
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes per baseline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--ppo_checkpoint",
        type=str,
        default=None,
        help="Path to trained PPO checkpoint (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baselines",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    main(args)
