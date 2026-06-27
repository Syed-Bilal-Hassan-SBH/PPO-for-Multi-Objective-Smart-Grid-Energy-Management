"""
Proximal Policy Optimization (PPO) Agent

Implements PPO with:
- Proper Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function clipping
- Gradient clipping
- KL divergence monitoring
- Entropy bonus

Based on: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

from src.models.networks import ActorCritic


class PPOAgent:
    """
    PPO agent with proper GAE and all improvements from the PPO paper.
    
    Improvements:
    - Learning rate decay schedule
    - Entropy coefficient decay schedule
    - Value function clipping
    - KL divergence monitoring
    - Comprehensive logging
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = 0.02,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        learning_rate_schedule: str = "linear",
        ent_coef_schedule: str = "linear",
        total_timesteps: int = 1000000
    ):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            learning_rate: Initial learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range (epsilon)
            clip_range_vf: Value function clip range (0.02 recommended)
            ent_coef: Initial entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Batch size for updates
            device: Device to use
            learning_rate_schedule: 'constant', 'linear', or 'exponential'
            ent_coef_schedule: 'constant', 'linear', or 'exponential'
            total_timesteps: Total training timesteps (for schedule)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf if clip_range_vf is not None else 0.02
        self.ent_coef_init = ent_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Schedule settings
        self.learning_rate_init = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.ent_coef_schedule = ent_coef_schedule
        self.total_timesteps = total_timesteps
        
        # Create policy network
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training statistics
        self.n_updates = 0
        self.num_timesteps = 0
        
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        This is the CORRECT implementation as per the PPO paper.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for next state
            
        Returns:
            (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def _update_learning_rate_schedule(self):
        """Update learning rate according to schedule."""
        progress = self.num_timesteps / max(1, self.total_timesteps)
        progress = min(progress, 1.0)
        
        if self.learning_rate_schedule == "linear":
            lr = self.learning_rate_init * (1.0 - progress)
        elif self.learning_rate_schedule == "exponential":
            lr = self.learning_rate_init * (0.5 ** progress)
        else:  # constant
            lr = self.learning_rate_init
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, 1e-6)  # Ensure non-zero learning rate
    
    def _update_entropy_schedule(self):
        """Update entropy coefficient according to schedule."""
        progress = self.num_timesteps / max(1, self.total_timesteps)
        progress = min(progress, 1.0)
        
        if self.ent_coef_schedule == "linear":
            self.ent_coef = self.ent_coef_init * (1.0 - progress)
        elif self.ent_coef_schedule == "exponential":
            self.ent_coef = self.ent_coef_init * (0.1 ** progress)
        else:  # constant
            self.ent_coef = self.ent_coef_init
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_values: np.ndarray,
        num_timesteps: int = 0
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            obs: Observations
            actions: Actions taken
            old_log_probs: Log probabilities of actions under old policy
            advantages: Advantage estimates
            returns: Return estimates
            old_values: Value estimates from old policy
            num_timesteps: Current number of timesteps (for schedules)
            
        Returns:
            Dictionary of training statistics
        """
        # Update number of timesteps
        if num_timesteps > 0:
            self.num_timesteps = num_timesteps
        else:
            self.num_timesteps += len(obs)
        
        # Update learning rate and entropy schedules
        self._update_learning_rate_schedule()
        self._update_entropy_schedule()
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        old_values_t = torch.FloatTensor(old_values).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        n_batches = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(obs))
            
            for start_idx in range(0, len(obs), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                obs_batch = obs_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                old_log_probs_batch = old_log_probs_t[batch_indices]
                advantages_batch = advantages_t[batch_indices]
                returns_batch = returns_t[batch_indices]
                old_values_batch = old_values_t[batch_indices]
                
                # Evaluate actions under current policy
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs_batch, actions_batch
                )
                
                # Compute ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Compute surrogate losses
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch
                
                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (IMPROVED from original)
                value_pred_clipped = old_values_batch + torch.clamp(
                    new_values - old_values_batch,
                    -self.clip_range_vf,
                    self.clip_range_vf
                )
                value_loss_1 = (new_values - returns_batch).pow(2)
                value_loss_2 = (value_pred_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_loss_1, value_loss_2).mean()
                
                # Entropy loss with dynamic coefficient
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    # Approximate KL divergence
                    kl = (old_log_probs_batch - new_log_probs).mean()
                    
                    # Clip fraction (fraction of samples where ratio was clipped)
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl.item()
                total_clip_fraction += clip_fraction.item()
                n_batches += 1
        
        self.n_updates += 1
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Return statistics (all metrics properly computed)
        return {
            'policy_loss': float(total_policy_loss / n_batches),
            'value_loss': float(total_value_loss / n_batches),
            'entropy': float(total_entropy / n_batches),
            'entropy_coef': float(self.ent_coef),
            'kl_divergence': float(total_kl / n_batches),
            'clip_fraction': float(total_clip_fraction / n_batches),
            'learning_rate': float(current_lr),
            'n_updates': self.n_updates,
            'progress': self.num_timesteps / self.total_timesteps
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy.
        
        Args:
            obs: Observation
            deterministic: If True, return mean action
            
        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if deterministic:
                action_mean, _, value = self.policy(obs_t)
                action = action_mean
                log_prob = 0.0  # Not used in deterministic mode
            else:
                action, log_prob, _, value = self.policy.get_action_and_value(obs_t)
            
            return action.cpu().numpy()[0], log_prob.cpu().item() if not deterministic else 0.0, value.cpu().item()
    
    def get_value(self, obs: np.ndarray) -> float:
        """
        Get value estimate for observation.
        
        Args:
            obs: Observation
            
        Returns:
            Value estimate
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.policy.get_value(obs_t)
            return value.cpu().item()
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_updates': self.n_updates
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_updates = checkpoint.get('n_updates', 0)
