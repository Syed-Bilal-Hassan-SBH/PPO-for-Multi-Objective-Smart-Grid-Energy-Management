"""
Safety-Constrained Environment Wrapper

Enforces hard constraints on battery SoC, power limits, and comfort bounds.
Integrates with reward function to provide penalties for constraint violations.
"""

import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym
from typing import Tuple, Dict, Any, Optional


class SafetyConstraints:
    """
    Defines safety constraints for smart grid energy management.
    """
    
    def __init__(
        self,
        battery_soc_min: float = 0.1,
        battery_soc_max: float = 0.9,
        max_power_per_building: float = 10.0,
        min_comfort_temp: float = 18.0,
        max_comfort_temp: float = 26.0,
        max_comfort_deviation: float = 2.0
    ):
        """
        Initialize constraints.
        
        Args:
            battery_soc_min: Minimum battery state of charge
            battery_soc_max: Maximum battery state of charge
            max_power_per_building: Max power draw per building (kW)
            min_comfort_temp: Minimum comfortable temperature
            max_comfort_temp: Maximum comfortable temperature
            max_comfort_deviation: Max deviation from comfort band
        """
        self.battery_soc_min = battery_soc_min
        self.battery_soc_max = battery_soc_max
        self.max_power = max_power_per_building
        self.min_comfort = min_comfort_temp
        self.max_comfort = max_comfort_temp
        self.max_deviation = max_comfort_deviation
    
    @property
    def soc_min(self):
        return self.battery_soc_min

    @property
    def soc_max(self):
        return self.battery_soc_max
    
    def check_battery_soc(self, soc: float) -> Tuple[bool, str]:
        """
        Check if battery SoC is within safe bounds.
        
        Args:
            soc: State of charge [0, 1]
            
        Returns:
            (is_safe, violation_message)
        """
        if soc < self.battery_soc_min:
            return False, f"Battery SoC too low: {soc:.3f} < {self.battery_soc_min}"
        if soc > self.battery_soc_max:
            return False, f"Battery SoC too high: {soc:.3f} > {self.battery_soc_max}"
        return True, ""
    
    def check_power_limit(self, power: float) -> Tuple[bool, str]:
        """
        Check if power draw is within limits.
        
        Args:
            power: Power draw (absolute value in kW)
            
        Returns:
            (is_safe, violation_message)
        """
        if np.abs(power) > self.max_power:
            return False, f"Power limit exceeded: {np.abs(power):.3f} > {self.max_power}"
        return True, ""
    
    def check_comfort(self, current_temp: float, setpoint_temp: float) -> Tuple[bool, str]:
        """
        Check if temperature is within comfort band.
        
        Args:
            current_temp: Current building temperature
            setpoint_temp: Target temperature setpoint
            
        Returns:
            (is_safe, violation_message)
        """
        if current_temp < self.min_comfort:
            return False, f"Temperature too cold: {current_temp:.1f} < {self.min_comfort}"
        if current_temp > self.max_comfort:
            return False, f"Temperature too hot: {current_temp:.1f} > {self.max_comfort}"
        
        deviation = np.abs(current_temp - setpoint_temp)
        if deviation > self.max_deviation:
            return False, f"Comfort deviation too high: {deviation:.1f} > {self.max_deviation}"
        
        return True, ""
    
    def check_all(
        self,
        state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, bool], Dict[str, str]]:
        """
        Check all constraints.
        
        Args:
            state: State dictionary with constraint-relevant info
            
        Returns:
            (all_safe, violations_dict, messages_dict)
        """
        violations = {}
        messages = {}
        
        # Battery constraints
        if 'battery_soc' in state:
            safe, msg = self.check_battery_soc(state['battery_soc'])
            violations['battery_soc'] = not safe
            if not safe:
                messages['battery_soc'] = msg
        
        # Power constraints
        if 'power_draw' in state:
            safe, msg = self.check_power_limit(state['power_draw'])
            violations['power'] = not safe
            if not safe:
                messages['power'] = msg
        
        # Comfort constraints
        if 'current_temp' in state and 'setpoint_temp' in state:
            safe, msg = self.check_comfort(state['current_temp'], state['setpoint_temp'])
            violations['comfort'] = not safe
            if not safe:
                messages['comfort'] = msg
        
        all_safe = all(not v for v in violations.values())
        
        return all_safe, violations, messages


class SafetyConstrainedWrapper(gym.Wrapper):
    """
    Wraps environment to enforce safety constraints and provide penalties.
    """
    
    def __init__(
        self,
        env: gym.Env,
        constraints: Optional[SafetyConstraints] = None,
        constraint_penalty: float = -100.0,
        action_clipping: bool = True,
        verbose: bool = False
    ):
        """
        Initialize safety wrapper.
        
        Args:
            env: Base environment
            constraints: SafetyConstraints object
            constraint_penalty: Reward penalty for constraint violation
            action_clipping: Whether to clip actions to safe range
            verbose: Whether to print constraint violations
        """
        super().__init__(env)
        
        self.constraints = constraints or SafetyConstraints()
        self.constraint_penalty = constraint_penalty
        self.action_clipping = action_clipping
        self.verbose = verbose
        
        # Statistics tracking
        self.constraint_violations = {
            'battery_soc': 0,
            'power': 0,
            'comfort': 0,
            'total': 0,
            'episodes': 0
        }
        self.total_steps = 0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment with safety checks.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, truncated, info)
        """
        # Clip action to safe range if enabled
        if self.action_clipping:
            action = np.clip(action, -1.0, 1.0)
        
        # Take step in environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Extract constraint-relevant info from observation/state
        constraint_state = self._extract_constraint_state(obs, info)
        
        # Check constraints
        all_safe, violations, messages = self.constraints.check_all(constraint_state)
        
        # Apply penalty if constraint violated
        if not all_safe:
            reward += self.constraint_penalty
            self.constraint_violations['total'] += 1
            for violation_type, violated in violations.items():
                if violated:
                    self.constraint_violations[violation_type] += 1
            
            if self.verbose:
                for msg in messages.values():
                    print(f"⚠️ Constraint violation: {msg}")
        
        # Add constraint info to info dict
        info['constraints'] = {
            'all_safe': all_safe,
            'violations': violations,
            'penalty_applied': not all_safe
        }
        
        self.total_steps += 1
        
        # Track episode end
        if done or truncated:
            self.constraint_violations['episodes'] += 1
            info['constraint_violation_rate'] = (
                self.constraint_violations['total'] / max(self.total_steps, 1)
            )
        
        return obs, reward, done, truncated, info
    
    def _extract_constraint_state(
        self,
        obs: np.ndarray,
        info: Dict
    ) -> Dict[str, Any]:
        """
        Extract constraint-relevant state from observation and info.
        
        This is environment-specific and may need customization for different
        energy management scenarios.
        
        Args:
            obs: Observation from environment
            info: Info dict from environment
            
        Returns:
            Dictionary with constraint state
        """
        state = {}
        
        # Try to extract from info dict (CityLearn-style)
        if 'battery_soc' in info:
            state['battery_soc'] = info['battery_soc']
        elif len(obs) > 0:
            # Heuristic: first few elements might be battery SoC
            # CityLearn typically puts battery info early in observation
            if obs[0] > 0 and obs[0] < 1:  # Likely a SoC value
                state['battery_soc'] = obs[0]
        
        if 'power_draw' in info:
            state['power_draw'] = info['power_draw']
        
        if 'current_temp' in info:
            state['current_temp'] = info['current_temp']
        
        if 'setpoint_temp' in info:
            state['setpoint_temp'] = info['setpoint_temp']
        
        return state
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and constraint tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode violation count
        self.constraint_violations['total'] = 0
        
        return obs, info
    
    def get_constraint_stats(self) -> Dict[str, float]:
        """
        Get constraint violation statistics.
        
        Returns:
            Dictionary with violation metrics
        """
        total_steps = max(self.total_steps, 1)
        
        return {
            'total_violations': self.constraint_violations['total'],
            'battery_violations': self.constraint_violations['battery_soc'],
            'power_violations': self.constraint_violations['power'],
            'comfort_violations': self.constraint_violations['comfort'],
            'episodes_trained': self.constraint_violations['episodes'],
            'violation_rate': self.constraint_violations['total'] / total_steps,
            'battery_violation_rate': self.constraint_violations['battery_soc'] / total_steps,
            'power_violation_rate': self.constraint_violations['power'] / total_steps,
            'comfort_violation_rate': self.constraint_violations['comfort'] / total_steps,
        }
    
    def reset_constraint_stats(self):
        """Reset all constraint statistics."""
        self.constraint_violations = {
            'battery_soc': 0,
            'power': 0,
            'comfort': 0,
            'total': 0,
            'episodes': 0
        }
        self.total_steps = 0


class MultiObjectiveSafetyOptimization:
    """
    Combines multi-objective rewards with safety constraints.
    
    Strategy: Use constraint violations as additional penalty in reward function
    while still allowing the agent to learn to satisfy objectives.
    """
    
    def __init__(
        self,
        cost_weight: float = 0.4,
        carbon_weight: float = 0.2,
        peak_weight: float = 0.2,
        comfort_weight: float = 0.2,
        constraint_penalty: float = -100.0
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            cost_weight: Weight for cost reduction
            carbon_weight: Weight for carbon reduction
            peak_weight: Weight for peak demand reduction
            comfort_weight: Weight for comfort maintenance
            constraint_penalty: Penalty for constraint violations
        """
        self.weights = {
            'cost': cost_weight,
            'carbon': carbon_weight,
            'peak': peak_weight,
            'comfort': comfort_weight
        }
        self.constraint_penalty = constraint_penalty
        
        # Normalize weights
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def compute_reward(
        self,
        cost: float,
        carbon: float,
        peak: float,
        comfort: float,
        constraint_violated: bool = False
    ) -> float:
        """
        Compute multi-objective reward with constraint safety.
        
        Args:
            cost: Cost metric (lower is better)
            carbon: Carbon metric (lower is better)
            peak: Peak demand metric (lower is better)
            comfort: Comfort metric (higher is better, or lower deviation)
            constraint_violated: Whether constraint was violated
            
        Returns:
            Aggregated reward
        """
        # Negate cost/carbon/peak (RL maximizes reward, but we want to minimize these)
        reward = (
            -self.weights['cost'] * cost +
            -self.weights['carbon'] * carbon +
            -self.weights['peak'] * peak +
            self.weights['comfort'] * comfort
        )
        
        # Apply constraint penalty
        if constraint_violated:
            reward += self.constraint_penalty
        
        return reward


def create_safe_environment(
    base_env: gym.Env,
    enable_constraints: bool = True,
    constraint_penalty: float = -100.0
) -> gym.Env:
    """
    Convenience function to create a safety-constrained environment.
    
    Args:
        base_env: Base environment
        enable_constraints: Whether to enable constraint checking
        constraint_penalty: Penalty for violations
        
    Returns:
        Wrapped environment
    """
    if enable_constraints:
        return SafetyConstrainedWrapper(
            base_env,
            constraint_penalty=constraint_penalty,
            action_clipping=True,
            verbose=False
        )
    return base_env
