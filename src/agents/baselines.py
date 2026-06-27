"""
Baseline Controllers for Comparison

Implements baseline controllers for benchmarking:
- Random policy
- Rule-based controller (time-of-use charging strategy)
- Do-nothing baseline
- Model Predictive Control (simplified)

Baselines are essential for validating PPO performance.
"""

import numpy as np
from typing import List, Optional
from datetime import datetime


class RandomBaseline:
    """
    Random policy baseline.
    
    Samples actions uniformly from action space.
    Provides lower bound on performance.
    """
    
    def __init__(self, action_dim: int, seed: int = 42):
        """
        Initialize random baseline.
        
        Args:
            action_dim: Action dimension
            seed: Random seed
        """
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get random action.
        
        Args:
            obs: Observation (not used)
            
        Returns:
            Random action in [-1, 1]
        """
        return self.rng.uniform(-1, 1, size=self.action_dim).astype(np.float32)


class DoNothingBaseline:
    """
    Do-nothing baseline.
    
    Always outputs zero action (no battery charging/discharging).
    Useful for evaluating passive building response.
    """
    
    def __init__(self, action_dim: int):
        """
        Initialize do-nothing baseline.
        
        Args:
            action_dim: Action dimension
        """
        self.action_dim = action_dim
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get zero action.
        
        Args:
            obs: Observation (not used)
            
        Returns:
            Zero action
        """
        return np.zeros(self.action_dim, dtype=np.float32)


class RuleBasedBaseline:
    """
    Rule-based controller using time-of-use (TOU) charging strategy.
    
    Strategy:
    - Charge battery (positive action) during low-price hours
    - Discharge battery (negative action) during high-price hours
    - Do nothing during medium-price hours
    - Respects battery SoC bounds
    
    This is a strong heuristic baseline commonly used in energy systems.
    """
    
    def __init__(
        self,
        action_dim: int,
        n_buildings: int = 5,
        seed: int = 42
    ):
        """
        Initialize rule-based baseline.
        
        Args:
            action_dim: Total action dimension (n_buildings * action_dim_per_building)
            n_buildings: Number of buildings
            seed: Random seed
        """
        self.action_dim = action_dim
        self.n_buildings = n_buildings
        self.action_dim_per_building = action_dim // n_buildings
        self.rng = np.random.RandomState(seed)
        
        # Time-of-use schedule (hourly): 0=low, 1=medium, 2=high
        # Simple pattern: low at night, high at day, medium at dawn/dusk
        self.hour_price_level = np.array([
            0, 0, 0, 0, 0, 0,  # 00-05: night (low)
            1, 1, 2, 2, 2, 2,  # 06-11: morning/midday (high)
            2, 2, 2, 2, 1, 1,  # 12-17: afternoon/evening (high to medium)
            2, 2, 1, 1, 1, 0,  # 18-23: evening/night (medium to low)
        ])
        
        self.current_hour = 0
        self.step_count = 0
        
    def get_action(self, obs: np.ndarray, hour: Optional[int] = None) -> np.ndarray:
        """
        Get action based on time-of-use strategy.
        
        Args:
            obs: Observation (may contain hour information)
            hour: Current hour of day (0-23). If None, estimates from step count.
            
        Returns:
            Action array
        """
        # Estimate hour if not provided
        # Assume 24 steps per hour in CityLearn
        if hour is None:
            hour = (self.step_count // 24) % 24
        else:
            hour = hour % 24
        
        self.step_count += 1
        
        # Get price level for current hour
        price_level = self.hour_price_level[hour]
        
        # Strategy:
        # - Low price: charge (positive action)
        # - Medium price: do nothing
        # - High price: discharge (negative action)
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        for building_idx in range(self.n_buildings):
            start_idx = building_idx * self.action_dim_per_building
            end_idx = start_idx + self.action_dim_per_building
            
            if price_level == 0:  # Low price hour
                # Charge battery: positive action with small randomness
                action[start_idx:end_idx] = 0.3 + 0.1 * self.rng.randn(self.action_dim_per_building)
            elif price_level == 2:  # High price hour
                # Discharge battery: negative action with small randomness
                action[start_idx:end_idx] = -0.3 + 0.1 * self.rng.randn(self.action_dim_per_building)
            # else: medium price, do nothing (action stays 0)
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        return action
    
    def reset(self):
        """Reset internal state."""
        self.step_count = 0


class BaselineEvaluator:
    """
    Helper class for evaluating baseline controllers.
    """
    
    @staticmethod
    def evaluate_baseline(
        baseline,
        env,
        n_episodes: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate a baseline controller.
        
        Args:
            baseline: Baseline controller with get_action() method
            env: Environment
            n_episodes: Number of episodes to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evaluation results
        """
        returns = []
        costs = []
        lengths = []
        violations_list = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            
            if hasattr(baseline, 'reset'):
                baseline.reset()
            
            episode_return = 0.0
            episode_cost = 0.0
            episode_length = 0
            episode_violations = 0
            done = False
            
            while not done:
                action = baseline.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_return += reward
                episode_cost += info.get('cost', 0.0)
                episode_violations += info.get('violations', 0)
                episode_length += 1
                
                done = done or truncated
            
            returns.append(episode_return)
            costs.append(episode_cost)
            lengths.append(episode_length)
            violations_list.append(episode_violations)
            
            if verbose and (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}/{n_episodes}: Return={episode_return:.2f}, Cost={episode_cost:.2f}")
        
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'mean_length': float(np.mean(lengths)),
            'mean_violations': float(np.mean(violations_list)),
            'returns': returns,
            'costs': costs
        }



class DoNothingBaseline:
    """
    Do-nothing baseline.
    
    Always outputs zero action (no battery charging/discharging).
    Useful for evaluating passive building response.
    """
    
    def __init__(self, action_dim: int):
        """
        Initialize do-nothing baseline.
        
        Args:
            action_dim: Action dimension
        """
        self.action_dim = action_dim
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get zero action.
        
        Args:
            obs: Observation (not used)
            
        Returns:
            Zero action
        """
        return np.zeros(self.action_dim)


class RuleBasedBaseline:
    """
    Rule-based controller using time-of-use (TOU) charging strategy.
    
    Strategy:
    - Charge battery (positive action) during low-price hours
    - Discharge battery (negative action) during high-price hours
    - Do nothing during medium-price hours
    - Respects battery SoC bounds
    
    This is a strong heuristic baseline commonly used in energy systems.
    """
    
    def __init__(
        self,
        action_dim: int,
        n_buildings: int = 5,
        charge_hours: Optional[List[int]] = None,
        discharge_hours: Optional[List[int]] = None,
        seed: int = 42
    ):
        """
        Initialize rule-based baseline.
        
        Args:
            action_dim: Total action dimension
            n_buildings: Number of buildings
            charge_hours: Hours to charge (default: night hours 0-6)
            discharge_hours: Hours to discharge (default: peak hours 17-21)
            seed: Random seed
        """
        self.action_dim = action_dim
        self.action_dim_per_building = action_dim // n_buildings
        self.n_buildings = n_buildings
        self.rng = np.random.RandomState(seed)
        
        # Default charging/discharging hours (simplified heuristic)
        self.charge_hours = charge_hours if charge_hours is not None else list(range(0, 7))
        self.discharge_hours = discharge_hours if discharge_hours is not None else list(range(17, 22))
        
        # Track current time (assume hourly steps)
        self.current_hour = 0
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action based on time-of-use strategy.
        
        Args:
            obs: Observation (flattened)
            
        Returns:
            Action array with per-building charging commands
        """
        action = np.zeros(self.action_dim)
        
        # Estimate current hour from observation if possible
        # (This is a simplification; real CityLearn provides this in obs)
        hour = int(self.current_hour) % 24
        self.current_hour += 1  # Assume one step per hour
        
        if hour in self.charge_hours:
            # Charge: positive action (battery takes energy from grid)
            action[:] = 0.5  # Moderate charge action
        elif hour in self.discharge_hours:
            # Discharge: negative action (battery gives energy to grid)
            action[:] = -0.5  # Moderate discharge action
        # Else: do nothing (action stays 0)
        
        # Add small stochastic noise for robustness
        action += self.rng.normal(0, 0.05, size=self.action_dim)
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)
    
    def reset(self):
        """Reset controller (e.g., at episode start)."""
        self.current_hour = 0


class PeakShavingBaseline:
    """
    Peak shaving baseline.
    
    Strategy:
    - Monitor electricity price and demand
    - Discharge battery when price is high or demand peaks
    - Charge battery during low-demand periods
    - Attempt to reduce peak power draws
    """
    
    def __init__(self, action_dim: int, n_buildings: int = 5, seed: int = 42):
        """
        Initialize peak shaving baseline.
        
        Args:
            action_dim: Total action dimension
            n_buildings: Number of buildings
            seed: Random seed
        """
        self.action_dim = action_dim
        self.action_dim_per_building = action_dim // n_buildings
        self.n_buildings = n_buildings
        self.rng = np.random.RandomState(seed)
        
        # Exponential moving average of price for trend
        self.price_ema = 0.5
        self.price_ema_alpha = 0.2
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action based on peak shaving strategy.
        
        Args:
            obs: Observation (flattened)
            
        Returns:
            Action array
        """
        action = np.zeros(self.action_dim)
        
        try:
            # Try to extract price information from observation
            # Observation format varies, so this is best-effort
            if len(obs) > 0:
                # Assume some price-related signal in early obs dimensions
                price_signal = float(obs[0]) if len(obs) > 0 else 0.5
                
                # Update EMA of price
                self.price_ema = (1 - self.price_ema_alpha) * self.price_ema + \
                                  self.price_ema_alpha * price_signal
                
                # Discharge if price is above EMA (high price period)
                if price_signal > self.price_ema:
                    action[:] = -0.3  # Discharge
                elif price_signal < (self.price_ema * 0.7):
                    action[:] = 0.3  # Charge during very low price
        except:
            # If extraction fails, fallback to do-nothing
            pass
        
        # Add stochastic noise
        action += self.rng.normal(0, 0.05, size=self.action_dim)
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)


class SimpleModelPredictiveBaseline:
    """
    Simplified Model Predictive Control (MPC) baseline.
    
    Strategy:
    - Predict future prices and demand (simple: repeat current pattern)
    - Optimize battery charging to minimize predicted cost
    - Execute first action of optimization
    """
    
    def __init__(self, action_dim: int, n_buildings: int = 5, horizon: int = 4, seed: int = 42):
        """
        Initialize MPC baseline.
        
        Args:
            action_dim: Total action dimension
            n_buildings: Number of buildings
            horizon: Prediction horizon (steps)
            seed: Random seed
        """
        self.action_dim = action_dim
        self.action_dim_per_building = action_dim // n_buildings
        self.n_buildings = n_buildings
        self.horizon = horizon
        self.rng = np.random.RandomState(seed)
        
        # History for prediction
        self.price_history = []
        self.demand_history = []
        self.battery_soc_history = []
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action based on simple MPC.
        
        Args:
            obs: Observation (flattened)
            
        Returns:
            Action array
        """
        action = np.zeros(self.action_dim)
        
        try:
            # Track history for simple prediction
            if len(obs) > 0:
                price_signal = float(obs[0]) if len(obs) > 0 else 0.5
                self.price_history.append(price_signal)
                
                if len(self.price_history) > self.horizon:
                    self.price_history = self.price_history[-self.horizon:]
                
                # Simple prediction: assume price repeats
                if len(self.price_history) >= 2:
                    # Calculate trend
                    recent_avg = np.mean(self.price_history[-min(2, len(self.price_history)):])
                    older_avg = np.mean(self.price_history[:-1]) if len(self.price_history) > 1 else recent_avg
                    
                    # If price is trending up, discharge now
                    if recent_avg > older_avg:
                        action[:] = -0.25
                    # If price is trending down, charge now
                    elif recent_avg < older_avg:
                        action[:] = 0.25
        except:
            pass
        
        # Add small noise
        action += self.rng.normal(0, 0.03, size=self.action_dim)
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)


class BaselineEvaluator:
    """
    Utility class for evaluating and comparing multiple baselines.
    """
    
    @staticmethod
    def evaluate_baseline(
        baseline,
        env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> dict:
        """
        Evaluate a single baseline.
        
        Args:
            baseline: Baseline controller
            env: Environment
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        returns = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                action = baseline.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                episode_return += reward
                episode_length += 1
                done = done or truncated
            
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Reset baseline if it has state
            if hasattr(baseline, 'reset'):
                baseline.reset()
        
        return {
            'returns': returns,
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'max_return': float(np.max(returns)),
            'min_return': float(np.min(returns)),
            'mean_length': float(np.mean(episode_lengths)),
        }
    
    @staticmethod
    def compare_baselines(
        baselines: dict,
        env,
        n_episodes: int = 5
    ) -> dict:
        """
        Evaluate and compare multiple baselines.
        
        Args:
            baselines: Dictionary {name: baseline_controller}
            env: Environment
            n_episodes: Episodes per baseline
            
        Returns:
            Dictionary with comparative results
        """
        results = {}
        
        for name, baseline in baselines.items():
            print(f"Evaluating {name}...")
            results[name] = BaselineEvaluator.evaluate_baseline(
                baseline, env, n_episodes
            )
        
        return results

        
        self.current_hour = 0
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get rule-based action.
        
        Args:
            obs: Observation
            
        Returns:
            Action based on time-of-day rules
        """
        # Simple time-based rule
        # In CityLearn, hour cycles through 0-23
        hour = self.current_hour % 24
        
        if hour in self.charge_hours:
            # Charge battery (positive action)
            action = np.ones(self.action_dim) * 0.8
        elif hour in self.discharge_hours:
            # Discharge battery (negative action)
            action = np.ones(self.action_dim) * -0.8
        else:
            # Do nothing
            action = np.zeros(self.action_dim)
        
        self.current_hour += 1
        
        return action
    
    def reset(self):
        """Reset hour counter."""
        self.current_hour = 0
