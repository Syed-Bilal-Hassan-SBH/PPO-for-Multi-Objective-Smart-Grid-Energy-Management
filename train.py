"""
Main Training Script for PPO on CityLearn

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seeding import set_seed
from src.agents.ppo import PPOAgent
from src.agents.baselines import RuleBasedBaseline
from src.agents.offline_rl import OfflineDataset, ImplicitQLearning, OfflineToOnlineTransfer
from src.envs.citylearn_wrapper import CityLearnWrapper
from src.training.trainer import Trainer


def load_citylearn_env(config: dict):
    """
    Load CityLearn environment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CityLearn environment
    """
    try:
        # Try to import and create CityLearn environment
        from citylearn import CityLearnEnv
        
        # Create base environment
        env = CityLearnEnv(
            config_path="schema.json"  # You may need to adjust this path
        )
        
        return env
    except ImportError:
        print("CityLearn not installed. Using dummy environment for demonstration.")
        return None


def create_dummy_env(n_buildings: int = 5, obs_dim_per_building: int = 15, action_dim_per_building: int = 1):
    """
    Create enhanced dummy environment with realistic smart grid dynamics.
    
    This environment simulates:
    - Time-of-day electricity pricing (TOU)
    - Carbon intensity variations
    - Building energy loads with daily patterns
    - Battery storage with realistic dynamics
    - Safety constraints (battery SoC limits)
    
    Args:
        n_buildings: Number of buildings
        obs_dim_per_building: Observation dimension per building
        action_dim_per_building: Action dimension per building
        
    Returns:
        Enhanced dummy gym environment with smart grid dynamics
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        import gym
        from gym import spaces
    import numpy as np
    
    class EnhancedDummyCityLearnEnv(gym.Env):
        """Enhanced dummy environment with realistic smart grid dynamics."""
        
        def __init__(self, n_buildings, obs_dim, action_dim):
            self.n_buildings = n_buildings
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            
            # Multi-building observation/action spaces (list of spaces)
            self.observation_space = [
                spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype='float32')
                for _ in range(n_buildings)
            ]
            self.action_space = [
                spaces.Box(low=-1, high=1, shape=(action_dim,), dtype='float32')
                for _ in range(n_buildings)
            ]
            
            self.step_count = 0
            self.max_steps = 8760  # One year, hourly timesteps
            
            # Smart grid parameters
            self.hour_of_day = 0
            self.day_of_year = 0
            
            # Battery state for each building (SoC: 0-1)
            self.battery_soc = np.ones(n_buildings) * 0.5
            self.battery_capacity = 10.0  # kWh
            
            # Baseline loads (kW) per building
            self.base_loads = np.random.uniform(2.0, 5.0, n_buildings)
            
            # TOU pricing tiers ($/kWh)
            self.prices = {
                'off_peak': 0.08,     # 11 PM - 7 AM
                'mid_peak': 0.12,     # 7 AM - 5 PM, 9 PM - 11 PM (non-peak hours)
                'on_peak': 0.25       # 5 PM - 9 PM (peak demand)
            }
            
            # Carbon intensity values (kg CO2/kWh)
            self.carbon_intensity = {
                'low': 0.2,      # Renewable-heavy hours
                'medium': 0.5,   # Mixed grid
                'high': 0.8      # Fossil-heavy hours
            }
        
        def _get_pricing_tier(self, hour):
            """Get pricing tier based on hour of day."""
            if 23 <= hour or hour < 7:
                return 'off_peak'
            elif 17 <= hour < 21:  # 5 PM - 9 PM
                return 'on_peak'
            else:
                return 'mid_peak'
        
        def _get_carbon_tier(self, hour):
            """Get carbon intensity tier based on hour."""
            # More renewables during day (solar), less at night
            if 10 <= hour < 16:  # Midday solar peak
                return 'low'
            elif 6 <= hour < 10 or 16 <= hour < 20:
                return 'medium'
            else:
                return 'high'
        
        def _get_load_pattern(self, hour, building_idx):
            """Get realistic building load pattern."""
            # Base load
            base = self.base_loads[building_idx]
            
            # Time-of-day variation
            if 6 <= hour < 9:  # Morning peak
                load_multiplier = 1.5
            elif 17 <= hour < 21:  # Evening peak
                load_multiplier = 1.8
            elif 0 <= hour < 6:  # Night (low)
                load_multiplier = 0.6
            else:
                load_multiplier = 1.0
            
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            load = base * load_multiplier * (1 + noise)
            
            return max(0.1, load)  # Ensure non-negative
        
        def reset(self):
            self.step_count = 0
            self.hour_of_day = 0
            self.day_of_year = 0
            
            # Reset battery to 50%
            self.battery_soc = np.ones(self.n_buildings) * 0.5
            
            # Generate initial observations
            obs = self._get_observations()
            info = {}
            return obs, info
        
        def _get_observations(self):
            """Generate realistic observations for each building."""
            obs_list = []
            
            for b in range(self.n_buildings):
                # Current hour (normalized)
                hour_norm = self.hour_of_day / 24.0
                
                # Day of year (normalized)
                day_norm = self.day_of_year / 365.0
                
                # Current price
                price_tier = self._get_pricing_tier(self.hour_of_day)
                price = self.prices[price_tier] / 0.25  # Normalize by max price
                
                # Current carbon intensity
                carbon_tier = self._get_carbon_tier(self.hour_of_day)
                carbon = self.carbon_intensity[carbon_tier] / 0.8  # Normalize
                
                # Building load
                load = self._get_load_pattern(self.hour_of_day, b)
                load_norm = load / 10.0  # Normalize
                
                # Battery SoC
                soc = self.battery_soc[b]
                
                # Combine into observation vector
                obs = np.array([
                    hour_norm,
                    day_norm,
                    price,
                    carbon,
                    load_norm,
                    soc,  # Current battery SoC
                    soc - 0.5,  # SoC deviation from target (0.5)
                    0.0,  # Net power (placeholder)
                    0.0,  # Grid power (placeholder)
                    np.sin(2 * np.pi * hour_norm),  # Cyclic time encoding
                    np.cos(2 * np.pi * hour_norm),
                    np.sin(2 * np.pi * day_norm),  # Seasonal encoding
                    np.cos(2 * np.pi * day_norm),
                    0.0,  # Padding
                    0.0   # Padding
                ], dtype='float32')
                
                obs_list.append(obs)
            
            return obs_list
        
        def step(self, actions):
            """
            Execute actions and return next state.
            
            Actions represent battery charge/discharge rate:
            - action > 0: discharge battery (provide power)
            - action < 0: charge battery (store power)
            - action ≈ 0: no battery operation
            """
            self.step_count += 1
            self.hour_of_day = (self.step_count % 24)
            self.day_of_year = (self.step_count // 24) % 365
            
            # Get current pricing and carbon
            price_tier = self._get_pricing_tier(self.hour_of_day)
            electricity_price = self.prices[price_tier]
            
            carbon_tier = self._get_carbon_tier(self.hour_of_day)
            carbon_intensity = self.carbon_intensity[carbon_tier]
            
            # Compute rewards for each building
            rewards = []
            total_violations = 0
            
            for b in range(self.n_buildings):
                # Get building load
                load = self._get_load_pattern(self.hour_of_day, b)
                
                # Battery action (clipping to valid range)
                battery_action = np.clip(actions[b][0], -1.0, 1.0)
                
                # Convert action to power (kW)
                # Positive = discharge, Negative = charge
                battery_power = battery_action * 3.0  # Max 3 kW charge/discharge
                
                # Update battery SoC
                # Charging increases SoC, discharging decreases SoC
                soc_change = -battery_power / self.battery_capacity  # Negative power = charging = increase SoC
                new_soc = self.battery_soc[b] + soc_change
                
                # Check safety constraints
                violation_penalty = 0.0
                if new_soc < 0.1 or new_soc > 0.9:
                    violation_penalty = -100.0  # Large penalty
                    total_violations += 1
                    new_soc = np.clip(new_soc, 0.1, 0.9)  # Enforce hard constraint
                
                self.battery_soc[b] = new_soc
                
                # Net power from grid (load minus battery discharge)
                net_power = load - battery_power
                net_power = max(0.0, net_power)  # Can't sell to grid in this simple model
                
                # Compute reward components
                # 1. Cost (minimize)
                cost = electricity_price * net_power
                cost_reward = -cost / 0.5  # Normalize (typical cost ~0.5)
                
                # 2. Carbon (minimize)
                carbon_emissions = carbon_intensity * net_power
                carbon_reward = -carbon_emissions / 1.0  # Normalize
                
                # 3. Peak demand (minimize peak hour usage)
                if price_tier == 'on_peak':
                    peak_penalty = -net_power / 5.0
                else:
                    peak_penalty = 0.0
                
                # 4. Battery health (keep near 50%, avoid deep cycling)
                health_penalty = -abs(new_soc - 0.5) * 2.0  # Penalize deviation
                health_penalty += -abs(battery_action) * 0.5  # Penalize frequent cycling
                
                # Combined reward (multi-objective)
                reward = (
                    0.4 * cost_reward +
                    0.2 * carbon_reward +
                    0.2 * peak_penalty +
                    0.2 * health_penalty +
                    violation_penalty
                )
                
                rewards.append(float(reward))
            
            # Check termination
            done = self.step_count >= self.max_steps
            truncated = False
            
            # Get next observations
            obs = self._get_observations()
            
            # Info dict
            info = {
                'cost': electricity_price * sum([self._get_load_pattern(self.hour_of_day, b) for b in range(self.n_buildings)]),
                'carbon': carbon_intensity * sum([self._get_load_pattern(self.hour_of_day, b) for b in range(self.n_buildings)]),
                'violations': total_violations,
                'hour': self.hour_of_day,
                'price_tier': price_tier,
                'carbon_tier': carbon_tier,
                'avg_soc': np.mean(self.battery_soc)
            }
            
            return obs, rewards, done, truncated, info
        
        def close(self):
            pass
    
    return EnhancedDummyCityLearnEnv(n_buildings, obs_dim_per_building, action_dim_per_building)


def run_offline_pretraining(
    env,
    agent,
    logger,
    n_episodes: int = 50,
    n_training_steps: int = 5000,
    batch_size: int = 32,
    device: str = "cpu"
):
    """
    Run offline pre-training using ImplicitQLearning (IQL).
    
    Args:
        env: Environment
        agent: PPO agent to transfer features to
        logger: Logger
        n_episodes: Episodes to collect with rule-based baseline
        n_training_steps: IQL training steps
        batch_size: Batch size for IQL training
        device: Device to use
    """
    logger.info("=" * 80)
    logger.info("OFFLINE PRE-TRAINING PHASE")
    logger.info("=" * 80)
    
    # Create rule-based baseline for data collection
    logger.info("Collecting offline trajectories with rule-based baseline...")
    baseline = RuleBasedBaseline(env.action_dim)
    dataset = OfflineDataset(obs_dim=env.obs_dim, action_dim=env.action_dim)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        trajectory = {"obs": [], "action": [], "reward": [], "next_obs": [], "done": []}
        
        while not done:
            action = baseline.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            trajectory["obs"].append(obs.copy())
            trajectory["action"].append(action)
            trajectory["reward"].append(reward)
            trajectory["next_obs"].append(next_obs.copy())
            trajectory["done"].append(done or truncated)
            
            obs = next_obs
        
        dataset.add_trajectory(trajectory)
        
        if (ep + 1) % 10 == 0:
            logger.info(f"  Collected {ep + 1}/{n_episodes} episodes")
    
    logger.info(f"Total transitions collected: {len(dataset)}")
    
    # Train IQL on collected data
    logger.info("Training Implicit Q-Learning on offline data...")
    
    iql = ImplicitQLearning(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        learning_rate=3e-4,
        tau=0.7,  # Conservative offline RL
        beta=1.0,
        device=device
    )
    
    for step in range(n_training_steps):
        batch = dataset.sample(batch_size)
        loss = iql.train_step(batch)
        
        if (step + 1) % 500 == 0:
            logger.info(f"  IQL step {step + 1}/{n_training_steps}, Loss: {loss:.4f}")
    
    # Transfer learned features to PPO agent
    logger.info("Transferring learned features to PPO agent...")
    transfer = OfflineToOnlineTransfer(
        iql_value_network=iql.value_network,
        ppo_agent=agent,
        transfer_layers=['value_function']
    )
    transfer.transfer_features()
    
    logger.info("Offline pre-training complete!")
    logger.info("=" * 80 + "\n")
    
    return dataset


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    seed = config.training.get('seed', 42)
    set_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create logger
    logger = Logger(
        log_dir=config.logging.get('log_dir', 'results/logs'),
        experiment_name="ppo_smart_grid",
        use_tensorboard=config.logging.get('tensorboard', True),
        verbose=1
    )

    
    logger.info("=" * 80)
    logger.info("PPO Smart Grid Training")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    
    # Create environment
    print("Creating environment...")
    try:
        base_env = load_citylearn_env(config)
        if base_env is None:
            base_env = create_dummy_env()
    except Exception as e:
        print(f"Error loading CityLearn: {e}")
        print("Using dummy environment instead")
        base_env = create_dummy_env()
    
    # Wrap environment
    env = CityLearnWrapper(
        base_env,
        normalize_obs=config.environment.get('normalize_obs', True),
        clip_obs=config.environment.get('clip_obs', 10.0),
        weight_cost=config.reward.get('weight_cost', 0.4),
        weight_carbon=config.reward.get('weight_carbon', 0.2),
        weight_peak=config.reward.get('weight_peak', 0.2),
        weight_health=config.reward.get('weight_health', 0.2),
        normalize_reward=config.reward.get('normalize_reward', True),
        clip_reward=config.reward.get('clip_reward', 10.0),
        use_safety_constraints=config.environment.get('use_safety_constraints', True)
    )

    
    logger.info(f"Environment observation dim: {env.obs_dim}")
    logger.info(f"Environment action dim: {env.action_dim}")
    
    # Create agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        learning_rate=float(config.training.get('learning_rate', 3e-4)),
        gamma=float(config.training.get('gamma', 0.99)),
        gae_lambda=float(config.training.get('gae_lambda', 0.95)),
        clip_range=float(config.training.get('clip_ratio', 0.2)),
        clip_range_vf=float(config.training.get('clip_range_vf', 0.02)),
        ent_coef=float(config.training.get('entropy_coef', 0.01)),
        vf_coef=float(config.training.get('value_coef', 0.5)),
        max_grad_norm=float(config.training.get('max_grad_norm', 0.5)),
        n_epochs=int(config.training.get('n_epochs', 10)),
        batch_size=int(config.training.get('batch_size', 64)),
        device=str(device),
        total_timesteps=int(config.training.get('total_timesteps', 1000000))
    )
    
    logger.info("Agent created successfully")
    
    # Optional: Run offline pre-training
    use_offline_pretraining = config.offline_pretraining.get('enabled', False)
    if use_offline_pretraining:
        run_offline_pretraining(
            env=env,
            agent=agent,
            logger=logger,
            n_episodes=config.training.get('offline_episodes', 50),
            n_training_steps=config.training.get('offline_steps', 5000),
            batch_size=config.training.get('offline_batch_size', 32),
            device=device
        )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        logger=logger,
        seed=seed,
        device=device
    )
    
    # Train
    logger.info("Starting training...")
    train_stats = trainer.train(
        total_timesteps=config.training.get('total_timesteps', 1000000),
        rollout_steps=config.training.get('rollout_steps', 2048),
        eval_interval=config.training.get('eval_interval', 10),
        save_interval=config.training.get('save_interval', 10),
        eval_episodes=config.training.get('eval_episodes', 5)
    )
    
    # Log final statistics
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Total steps: {train_stats['total_steps']}")
    logger.info(f"Total episodes: {train_stats['total_episodes']}")
    logger.info(f"Mean return: {train_stats['mean_return']:.2f} ± {train_stats['std_return']:.2f}")
    
    # Cleanup
    env.close()
    logger.close()


if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Train PPO on CityLearn")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    main(args)
