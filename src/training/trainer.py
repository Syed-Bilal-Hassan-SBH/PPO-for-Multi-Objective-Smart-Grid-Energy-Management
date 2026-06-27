"""
Training Loop for PPO with CityLearn

Handles:
- Trajectory collection
- PPO updates
- Evaluation
- Checkpointing
- Logging
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm


class Trainer:
    """Main training loop for PPO."""
    
    def __init__(self, env, agent, logger, seed: int = 42, device: str = "cpu"):
        """
        Initialize trainer.
        
        Args:
            env: Training environment
            agent: PPO agent
            logger: Logger for metrics
            seed: Random seed
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.agent = agent
        self.logger = logger
        self.seed = seed
        self.device = torch.device(device)
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_returns = []
    
    def train(
        self,
        total_timesteps: int = 1000000,
        rollout_steps: int = 2048,
        eval_interval: int = 10,
        save_interval: int = 10,
        eval_episodes: int = 5
    ) -> Dict:
        """
        Main training function.
        
        Args:
            total_timesteps: Total environment steps to train
            rollout_steps: Steps collected before PPO update
            eval_interval: Evaluate every N updates
            save_interval: Save checkpoint every N updates
            eval_episodes: Number of episodes for evaluation
            
        Returns:
            Dictionary with training statistics
        """
        obs, info = self.env.reset()
        
        update_num = 0
        returns = []
        
        # Initialize progress bar
        pbar = tqdm(total=total_timesteps, desc="Training PPO", unit="steps")
        
        while self.total_steps < total_timesteps:
            # 1. COLLECT ROLLOUT
            rollout_data = self._collect_rollout(obs, rollout_steps)
            self.total_steps += rollout_steps
            pbar.update(rollout_steps)
            update_num += 1
            
            # 2. COMPUTE GAE
            last_value = self.agent.get_value(rollout_data['next_obs'])
            advantages, returns_gae = self.agent.compute_gae(
                rollout_data['rewards'],
                rollout_data['values'],
                rollout_data['dones'],
                last_value
            )
            
            # 3. UPDATE AGENT
            update_stats = self.agent.update(
                rollout_data['obs'],
                rollout_data['actions'],
                rollout_data['log_probs'],
                advantages,
                returns_gae,
                rollout_data['values']
            )
            
            # 4. LOG TRAINING
            self.logger.log_dict(update_stats, step=self.total_steps)
            self.logger.log_scalar('total_episodes', self.total_episodes, step=self.total_steps)
            
            # 5. EVALUATE PERIODICALLY
            if update_num % eval_interval == 0:
                eval_return = self._evaluate(n_episodes=eval_episodes)
                self.logger.log_scalar('eval_return', eval_return, step=self.total_steps)
                returns.append(eval_return)
                
                self.logger.info(
                    f"Update {update_num} | "
                    f"Steps: {self.total_steps/1e6:.1f}M | "
                    f"Eval Return: {eval_return:.2f} | "
                    f"Episodes: {self.total_episodes}"
                )
            
            # 6. SAVE CHECKPOINT
            if update_num % save_interval == 0:
                checkpoint_dir = self.logger.experiment_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = checkpoint_dir / f"model_{self.total_steps}.pt"
                self.agent.save(str(checkpoint_path))
            
            # Move to next rollout
            obs = rollout_data['next_obs']
            self.total_episodes += rollout_data['episodes_collected']
        
        # Final statistics
        pbar.close()
        
        final_stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'mean_return': np.mean(returns) if returns else 0.0,
            'std_return': np.std(returns) if returns else 0.0,
        }
        
        return final_stats
    
    def _collect_rollout(self, initial_obs: np.ndarray, num_steps: int) -> Dict:
        """
        Collect trajectory data for one rollout.
        
        Args:
            initial_obs: Starting observation
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary with trajectory data
        """
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        log_prob_buffer = []
        reward_info_buffer = []
        filter_info_buffer = []
        
        obs = initial_obs
        episodes_collected = 0
        
        for step in range(num_steps):
            # Store observation
            obs_buffer.append(obs.copy())
            
            # Get action from policy
            action, log_prob, value = self.agent.get_action(obs, deterministic=False)

            # Safety filter (environment may project action into safe set)
            if hasattr(self.env, 'filter_actions'):
                try:
                    filtered_flat_action, filter_info = self.env.filter_actions(action, obs)
                    action_to_step = filtered_flat_action
                    filter_info_buffer.append(filter_info)
                except Exception:
                    action_to_step = action
                    filter_info_buffer.append({'filtered': False})
            else:
                action_to_step = action
                filter_info_buffer.append({'filtered': False})
            
            action_buffer.append(action_to_step)
            log_prob_buffer.append(log_prob)
            value_buffer.append(value)
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action_to_step)
            
            reward_buffer.append(reward)
            reward_info_buffer.append(info)
            done_buffer.append(done or truncated)
            
            # Handle episode end
            if done or truncated:
                obs, _ = self.env.reset()
                episodes_collected += 1
        
        return {
            'obs': np.array(obs_buffer, dtype=np.float32),
            'actions': np.array(action_buffer, dtype=np.float32),
            'rewards': np.array(reward_buffer, dtype=np.float32),
            'dones': np.array(done_buffer, dtype=bool),
            'values': np.array(value_buffer, dtype=np.float32),
            'log_probs': np.array(log_prob_buffer, dtype=np.float32),
            'next_obs': obs,
            'reward_info': reward_info_buffer,
            'filter_info': filter_info_buffer,
            'episodes_collected': episodes_collected
        }
    
    def _evaluate(self, n_episodes: int = 5, deterministic: bool = True) -> float:
        """
        Evaluate agent on environment.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use mean action (vs sampling)
            
        Returns:
            Mean return over episodes
        """
        returns = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_return = 0.0
            
            while not done:
                action, _, _ = self.agent.get_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                ep_return += reward
                done = done or truncated
            
            returns.append(ep_return)
        
        return float(np.mean(returns))
