"""
Neural Network Architectures for PPO

Implements Actor-Critic networks with proper initialization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def init_weights(module: nn.Module, gain: float = 1.0):
    """
    Orthogonal initialization for neural network weights.
    
    Args:
        module: Neural network module
        gain: Scaling factor for initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Implements:
    - Actor: Outputs mean and log_std for Gaussian policy
    - Critic: Outputs state value estimate
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        log_std_init: float = 0.0
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('tanh' or 'relu')
            log_std_init: Initial value for log std
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Choose activation
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build actor network (policy)
        actor_layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn()
            ])
            prev_dim = hidden_dim
        
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # Learnable log std (state-independent)
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Build critic network (value function)
        critic_layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn()
            ])
            prev_dim = hidden_dim
        
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights with orthogonal initialization
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Re-initialize output layers with smaller gain
        init_weights(self.actor_mean[-1], gain=0.01)
        init_weights(self.critic[-1], gain=1.0)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            obs: Observations
            
        Returns:
            (action_mean, action_std, state_value)
        """
        # Actor
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic
        state_value = self.critic(obs)
        
        return action_mean, action_std, state_value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            obs: Observations
            
        Returns:
            State values
        """
        return self.critic(obs)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observations
            action: Actions (if None, sample from policy)
            
        Returns:
            (action, log_prob, entropy, value)
        """
        action_mean, action_std, value = self.forward(obs)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)
