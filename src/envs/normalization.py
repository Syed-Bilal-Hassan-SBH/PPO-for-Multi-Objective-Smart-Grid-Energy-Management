"""
State and Reward Normalization

Implements running mean/std normalization for observations and rewards.
Critical for stable PPO training.
"""

import numpy as np
import torch


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, shape=(), epsilon=1e-4):
        """
        Initialize running statistics.
        
        Args:
            shape: Shape of the data
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x):
        """
        Update statistics with new data.
        
        Args:
            x: New data (can be batched)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Update from batch statistics.
        
        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation:
    """
    Normalize observations using running mean and std.
    """
    
    def __init__(self, shape, clip=10.0):
        """
        Initialize observation normalizer.
        
        Args:
            shape: Shape of observations
            clip: Clip normalized observations to [-clip, clip]
        """
        self.rms = RunningMeanStd(shape=shape)
        self.clip = clip
        
    def __call__(self, obs, update=True):
        """
        Normalize observation.
        
        Args:
            obs: Observation to normalize
            update: Whether to update statistics
            
        Returns:
            Normalized observation
        """
        if update:
            self.rms.update(np.array([obs]))
        
        normalized = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip)


class NormalizeReward:
    """
    Normalize rewards using running std (not mean).
    """
    
    def __init__(self, gamma=0.99, clip=10.0):
        """
        Initialize reward normalizer.
        
        Args:
            gamma: Discount factor
            clip: Clip normalized rewards to [-clip, clip]
        """
        self.return_rms = RunningMeanStd(shape=())
        self.returns = 0.0
        self.gamma = gamma
        self.clip = clip
        
    def __call__(self, reward, update=True):
        """
        Normalize reward.
        
        Args:
            reward: Reward to normalize
            update: Whether to update statistics
            
        Returns:
            Normalized reward
        """
        self.returns = reward + self.gamma * self.returns
        
        if update:
            self.rms.update(np.array([self.returns]))
        
        normalized = reward / np.sqrt(self.return_rms.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip)
    
    def reset(self):
        """Reset returns (call at episode end)."""
        self.returns = 0.0
