"""
Offline Reinforcement Learning Module

Implements Implicit Q-Learning (IQL) for offline pre-training.
Enables offline-to-online transfer learning for faster convergence.

Based on: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (2021)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from collections import deque


class OfflineDataset:
    """
    Dataset for offline RL training.
    Collects trajectories from a behavior policy (rule-based or random).
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize offline dataset.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.size = 0
        
        # Storage
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.returns = None
        
    def add_trajectory(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray
    ):
        """
        Add trajectory to dataset.
        
        Args:
            observations: Trajectory observations
            actions: Trajectory actions
            rewards: Trajectory rewards
            dones: Trajectory done flags
        """
        trajectory_length = len(observations) - 1  # Exclude last obs (next_obs only)
        
        if self.observations is None:
            # Initialize storage
            self.observations = np.zeros((self.capacity, observations.shape[-1]), dtype=np.float32)
            self.actions = np.zeros((self.capacity, actions.shape[-1]), dtype=np.float32)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.next_observations = np.zeros((self.capacity, observations.shape[-1]), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=bool)
            self.returns = np.zeros(self.capacity, dtype=np.float32)
        
        # Add transitions
        for t in range(trajectory_length):
            idx = self.size % self.capacity
            
            self.observations[idx] = observations[t]
            self.actions[idx] = actions[t]
            self.rewards[idx] = rewards[t]
            self.next_observations[idx] = observations[t + 1]
            self.dones[idx] = dones[t]
            
            # Compute return (cumulative discounted reward)
            G_t = 0
            for tau in range(t, trajectory_length):
                G_t += (0.99 ** (tau - t)) * rewards[tau]
            self.returns[idx] = G_t
            
            self.size += 1
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch from dataset.
        
        Args:
            batch_size: Batch size
            
        Returns:
            (observations, actions, rewards, next_observations, dones)
        """
        indices = np.random.randint(0, min(self.size, self.capacity), size=batch_size)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get all data in dataset."""
        n = min(self.size, self.capacity)
        return (
            self.observations[:n],
            self.actions[:n],
            self.rewards[:n],
            self.next_observations[:n],
            self.dones[:n],
            self.returns[:n]
        )


class ValueNetwork(nn.Module):
    """Value network for IQL."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QLearner(nn.Module):
    """Q-network for IQL."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


class ImplicitQLearning:
    """
    Implicit Q-Learning (IQL) for offline RL.
    
    Key idea: Learn policy that avoids out-of-distribution actions using
    advantage-weighted behavior cloning.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        tau: float = 0.7,  # Temperature parameter for advantage weighting
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        """
        Initialize IQL.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            tau: Temperature for advantage weighting (0=no weighting, 1=full weighting)
            gamma: Discount factor
            device: Device to use
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Networks
        self.q_network = QLearner(obs_dim, action_dim).to(self.device)
        self.q_target = QLearner(obs_dim, action_dim).to(self.device)
        self.value_network = ValueNetwork(obs_dim).to(self.device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.v_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Copy target
        self._soft_update(tau=1.0)
    
    def _soft_update(self, tau: float = 0.005):
        """Soft update target network."""
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        batch_size: int = 256,
        num_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Train IQL on offline data.
        
        Args:
            obs: Observations
            actions: Actions
            rewards: Rewards
            next_obs: Next observations
            dones: Done flags
            batch_size: Batch size
            num_steps: Number of training steps
            
        Returns:
            Training statistics
        """
        stats = {'q_loss': [], 'v_loss': []}
        
        for step in range(num_steps):
            # Sample batch
            indices = np.random.randint(0, len(obs), size=batch_size)
            
            obs_batch = torch.FloatTensor(obs[indices]).to(self.device)
            actions_batch = torch.FloatTensor(actions[indices]).to(self.device)
            rewards_batch = torch.FloatTensor(rewards[indices]).to(self.device)
            next_obs_batch = torch.FloatTensor(next_obs[indices]).to(self.device)
            dones_batch = torch.FloatTensor(dones[indices]).to(self.device)
            
            # ===== Train Q-network =====
            with torch.no_grad():
                next_v = self.value_network(next_obs_batch)
                target_q = rewards_batch.unsqueeze(-1) + self.gamma * next_v * (1 - dones_batch.unsqueeze(-1))
            
            current_q = self.q_network(obs_batch, actions_batch)
            q_loss = ((current_q - target_q) ** 2).mean()
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            # ===== Train Value Network =====
            # Advantage-weighted value loss
            with torch.no_grad():
                q_values = self.q_network(obs_batch, actions_batch)
                advantage = q_values - self.value_network(obs_batch)
                # Weight by advantage: exp(tau * advantage)
                advantage_weights = torch.exp(self.tau * advantage)
                advantage_weights = advantage_weights / (advantage_weights.max() + 1e-8)  # Normalize
            
            v_pred = self.value_network(obs_batch)
            v_loss = (advantage_weights * ((v_pred - q_values) ** 2)).mean()
            
            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()
            
            # Soft update target
            self._soft_update(tau=0.005)
            
            stats['q_loss'].append(q_loss.item())
            stats['v_loss'].append(v_loss.item())
        
        return {
            'q_loss': np.mean(stats['q_loss']),
            'v_loss': np.mean(stats['v_loss'])
        }
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.value_network(obs_t).cpu().item()
    
    def get_policy_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract learned features from value network.
        Useful for transfer learning to PPO.
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            # Get activations from first hidden layer
            hidden = torch.relu(self.value_network.net[0](obs_t))
            return hidden.cpu().numpy()[0]


class OfflineToOnlineTransfer:
    """
    Transfer learning from offline IQL to online PPO.
    """
    
    def __init__(self, iql_agent: ImplicitQLearning, ppo_agent):
        """
        Initialize transfer wrapper.
        
        Args:
            iql_agent: Trained offline IQL agent
            ppo_agent: PPO agent to transfer to
        """
        self.iql = iql_agent
        self.ppo = ppo_agent
    
    def transfer_features(self):
        """
        Transfer learned features from IQL to PPO.
        
        Strategy: Initialize PPO policy network with representations
        learned by IQL value network.
        """
        # Get IQL value network weights
        iql_weights = self.iql.value_network.net[0].weight.data.clone()
        iql_bias = self.iql.value_network.net[0].bias.data.clone()
        
        # Transfer to PPO actor network (first layer)
        if self.ppo.policy.actor_mean[0].weight.shape == iql_weights.shape:
            self.ppo.policy.actor_mean[0].weight.data.copy_(iql_weights)
            self.ppo.policy.actor_mean[0].bias.data.copy_(iql_bias)
    
    def get_initialization_hint(self) -> Dict:
        """
        Get learned value function as initialization hint for PPO.
        """
        return {
            'learned_features': True,
            'q_value_estimates': 'available'
        }
