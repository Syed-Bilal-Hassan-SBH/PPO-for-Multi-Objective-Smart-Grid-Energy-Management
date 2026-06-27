"""
Evaluation Script for Trained PPO Agent

Compare trained agent against baselines and analyze performance.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seeding import set_seed
from src.agents.ppo import PPOAgent
from src.agents.baselines import RandomController, DoNothingController, RuleBasedController, BaselineEvaluator
from src.envs.citylearn_wrapper import CityLearnWrapper


def evaluate_agent(
    agent: PPOAgent,
    env: CityLearnWrapper,
    n_episodes: int = 10,
    deterministic: bool = True
):
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained PPO agent
        env: Environment
        n_episodes: Number of evaluation episodes
        deterministic: Use mean action
        
    Returns:
        Dictionary with evaluation results
    """
    returns = []
    costs = []
    carbons = []
    peaks = []
    healths = []
    violations_list = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        
        while not done:
            action, _ = agent.select_action(
                obs, training=False, deterministic=deterministic
            )
            obs, reward, done, info = env.step(action)
            episode_return += reward
        
        returns.append(episode_return)
        
        # Extract metrics
        metrics = env.get_episode_metrics()
        costs.append(metrics.get("total_cost", 0.0))
        carbons.append(metrics.get("total_carbon", 0.0))
        peaks.append(metrics.get("peak_demand", 0.0))
        healths.append(metrics.get("avg_battery_health", 1.0))
        violations_list.append(metrics.get("violations", 0))
    
    return {
        "agent_type": "PPO",
        "n_episodes": n_episodes,
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "max_return": float(np.max(returns)),
        "min_return": float(np.min(returns)),
        "costs": costs,
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "carbons": carbons,
        "mean_carbon": float(np.mean(carbons)),
        "peaks": peaks,
        "mean_peak": float(np.mean(peaks)),
        "healths": healths,
        "mean_health": float(np.mean(healths)),
        "violations": float(np.mean(violations_list))
    }


def evaluate_baselines(
    env: CityLearnWrapper,
    n_episodes: int = 10
):
    """
    Evaluate baseline controllers.
    
    Args:
        env: Environment
        n_episodes: Episodes per baseline
        
    Returns:
        Dictionary with baseline results
    """
    baselines = {}
    
    # Random baseline
    print("Evaluating random controller...")
    random_ctrl = RandomController(env.action_space, seed=42)
    random_returns = []
    random_costs = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        
        while not done:
            action = random_ctrl.select_action(obs)
            obs, reward, done, info = env.step(action)
            ep_return += reward
        
        random_returns.append(ep_return)
        metrics = env.get_episode_metrics()
        random_costs.append(metrics.get("total_cost", 0.0))
    
    baselines["random"] = {
        "agent_type": "Random",
        "mean_return": float(np.mean(random_returns)),
        "std_return": float(np.std(random_returns)),
        "mean_cost": float(np.mean(random_costs)),
    }
    
    # Do-nothing baseline
    print("Evaluating do-nothing controller...")
    nothing_ctrl = DoNothingController(env.action_space)
    nothing_returns = []
    nothing_costs = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        
        while not done:
            action = nothing_ctrl.select_action(obs)
            obs, reward, done, info = env.step(action)
            ep_return += reward
        
        nothing_returns.append(ep_return)
        metrics = env.get_episode_metrics()
        nothing_costs.append(metrics.get("total_cost", 0.0))
    
    baselines["do_nothing"] = {
        "agent_type": "Do-Nothing",
        "mean_return": float(np.mean(nothing_returns)),
        "std_return": float(np.std(nothing_returns)),
        "mean_cost": float(np.mean(nothing_costs)),
    }
    
    # Rule-based baseline
    print("Evaluating rule-based controller...")
    rule_ctrl = RuleBasedController(env.action_space)
    rule_returns = []
    rule_costs = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        
        while not done:
            action = rule_ctrl.select_action(obs)
            obs, reward, done, info = env.step(action)
            ep_return += reward
        
        rule_returns.append(ep_return)
        metrics = env.get_episode_metrics()
        rule_costs.append(metrics.get("total_cost", 0.0))
    
    baselines["rule_based"] = {
        "agent_type": "Rule-Based",
        "mean_return": float(np.mean(rule_returns)),
        "std_return": float(np.std(rule_returns)),
        "mean_cost": float(np.mean(rule_costs)),
    }
    
    return baselines


def main(args):
    """Main evaluation function."""
    
    # Load configuration
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logger
    logger = Logger(
        log_dir=Path(config.logging.get('log_dir', 'results/logs')),
        checkpoint_dir=Path(config.logging.get('checkpoint_dir', 'results/checkpoints')),
        plot_dir=Path(config.logging.get('plot_dir', 'results/plots')),
        use_tensorboard=False
    )
    
    logger.info("=" * 80)
    logger.info("PPO Smart Grid - Agent Evaluation")
    logger.info("=" * 80)
    
    # Load trained agent
    if not args.checkpoint:
        logger.error("--checkpoint argument required")
        return
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Create dummy environment for testing
    import gym
    from gym import spaces
    
    class DummyEnv(gym.Env):
        def __init__(self):
            self.obs_dim = 15
            self.action_dim = 5
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(self.obs_dim,), dtype='float32'
            )
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(self.action_dim,), dtype='float32'
            )
            self.step_count = 0
            self.max_steps = 1000
        
        def reset(self):
            self.step_count = 0
            return self.observation_space.sample().astype('float32')
        
        def step(self, action):
            self.step_count += 1
            obs = self.observation_space.sample().astype('float32')
            reward = float(np.random.randn())
            done = self.step_count >= self.max_steps
            
            info = {
                "electricity_cost": np.random.uniform(0, 1),
                "carbon_intensity": np.random.uniform(0, 1),
                "peak_demand": np.random.uniform(0, 1),
                "battery_health": np.random.uniform(0.7, 1.0)
            }
            
            return obs, reward, done, info
        
        def render(self, mode='human'):
            pass
        
        def close(self):
            pass
    
    # Create environment
    base_env = DummyEnv()
    env = CityLearnWrapper(base_env)
    
    # Create agent
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device
    )
    
    # Load weights
    agent.load(str(checkpoint_path))
    logger.info("Checkpoint loaded successfully")
    
    # Evaluate agent
    logger.info(f"Evaluating agent on {args.episodes} episodes...")
    ppo_results = evaluate_agent(env, agent, n_episodes=args.episodes)
    
    logger.info(f"PPO Results:")
    logger.info(f"  Mean Return: {ppo_results['mean_return']:.2f} ± {ppo_results['std_return']:.2f}")
    logger.info(f"  Mean Cost: ${ppo_results['mean_cost']:.2f} ± ${ppo_results['std_cost']:.2f}")
    logger.info(f"  Mean Peak: {ppo_results['mean_peak']:.2f}")
    logger.info(f"  Violations: {ppo_results['violations']:.0f}")
    
    # Evaluate baselines if requested
    results = {"ppo": ppo_results}
    
    if args.baselines:
        logger.info("\nEvaluating baselines...")
        baseline_results = evaluate_baselines(env, n_episodes=args.episodes)
        results.update(baseline_results)
        
        logger.info("\nBaseline Comparison:")
        logger.info("-" * 80)
        
        for name, baseline in baseline_results.items():
            improvement = (
                (ppo_results['mean_return'] - baseline['mean_return']) /
                (abs(baseline['mean_return']) + 1e-6) * 100
            )
            logger.info(f"{baseline['agent_type']}:")
            logger.info(f"  Return: {baseline['mean_return']:.2f} ± {baseline['std_return']:.2f}")
            logger.info(f"  PPO improvement: {improvement:.1f}%")
    
    # Save results
    if args.save_results:
        results_path = Path("results/evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, (list, np.ndarray)):
                        results_json[key][k] = [float(x) for x in v]
                    else:
                        results_json[key][k] = v
            else:
                results_json[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_path}")
    
    logger.info("=" * 80)
    logger.info("Evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained agent checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Compare against baselines"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON"
    )
    
    args = parser.parse_args()
    main(args)
