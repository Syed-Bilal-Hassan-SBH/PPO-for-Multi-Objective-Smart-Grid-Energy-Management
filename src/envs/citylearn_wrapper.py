"""
CityLearn Environment Wrapper

Wraps CityLearn environment with:
- Centralized control (concatenates all building obs/actions)
- Proper observation/action handling
- State normalization
- Episode statistics tracking
- Multi-objective reward composition
- Safety constraints integration
"""

import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym
from typing import Tuple, List, Dict, Any

from src.envs.normalization import NormalizeObservation
from src.envs.safety_constraints import SafetyConstraints
from src.envs.reward_extractor import CityLearnMetricExtractor


class CityLearnWrapper(gym.Wrapper):
    """
    Wrapper for CityLearn environment with centralized control and multi-objective rewards.
    """
    
    def __init__(
        self,
        env,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        weight_cost: float = 0.4,
        weight_carbon: float = 0.2,
        weight_peak: float = 0.2,
        weight_health: float = 0.2,
        normalize_reward: bool = True,
        clip_reward: float = 10.0,
        use_safety_constraints: bool = True
    ):
        """
        Initialize wrapper.
        
        Args:
            env: CityLearn environment
            normalize_obs: Whether to normalize observations
            clip_obs: Clip normalized observations to [-clip, clip]
            weight_cost: Weight for cost objective
            weight_carbon: Weight for carbon objective
            weight_peak: Weight for peak demand objective
            weight_health: Weight for battery health objective
            normalize_reward: Whether to normalize rewards
            clip_reward: Clip rewards to [-clip_reward, clip_reward]
            use_safety_constraints: Whether to apply safety constraints
        """
        super().__init__(env)
        
        # Get dimensions
        self.num_buildings = len(env.observation_space)
        self.obs_dim_per_building = env.observation_space[0].shape[0]
        self.action_dim_per_building = env.action_space[0].shape[0]
        
        # Total dimensions for centralized control
        self.total_obs_dim = self.num_buildings * self.obs_dim_per_building
        self.total_action_dim = self.num_buildings * self.action_dim_per_building
        
        # Create normalizer
        self.normalize_obs = normalize_obs
        if normalize_obs:
            self.obs_normalizer = NormalizeObservation(
                shape=(self.total_obs_dim,),
                clip=clip_obs
            )
        
        # Multi-objective reward weights
        self.weight_cost = weight_cost
        self.weight_carbon = weight_carbon
        self.weight_peak = weight_peak
        self.weight_health = weight_health
        self.normalize_reward = normalize_reward
        self.clip_reward = clip_reward
        self.use_safety_constraints = use_safety_constraints
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.episode_cost = 0.0
        self.episode_violations = 0
        
        # Safety constraints
        self.safety_constraints = SafetyConstraints(
            battery_soc_min=0.1,
            battery_soc_max=0.9,
            max_power_per_building=10.0,
            min_comfort_temp=18.0,
            max_comfort_temp=26.0,
            max_comfort_deviation=2.0
        ) if use_safety_constraints else None
        
        # Metric extractor for true multi-objective rewards
        self.metric_extractor = CityLearnMetricExtractor(num_buildings=self.num_buildings)
        
        # Expose dimensions as properties
        self.obs_dim = self.total_obs_dim
        self.action_dim = self.total_action_dim
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.
        
        Returns:
            Flattened observation and info dict
        """
        result = self.env.reset(**kwargs)
        
        # Handle different return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        # Flatten observations from all buildings
        flat_obs = self._flatten_obs(obs)
        
        # Normalize if enabled
        if self.normalize_obs:
            flat_obs = self.obs_normalizer(flat_obs, update=False)
        
        # Reset episode stats
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_cost = 0.0
        self.episode_violations = 0
        
        return flat_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment with multi-objective reward.
        
        Args:
            action: Flattened action array
            
        Returns:
            (observation, reward, done, truncated, info)
        """
        # Split action for each building
        action_list = self._split_action(action)
        
        # Step environment
        result = self.env.step(action_list)
        
        # Handle different return formats
        if len(result) == 5:
            next_obs, reward, done, truncated, info = result
        else:
            next_obs, reward, done, info = result
            truncated = False
        
        # Flatten observations
        flat_obs = self._flatten_obs(next_obs)
        
        # Normalize if enabled
        if self.normalize_obs:
            flat_obs = self.obs_normalizer(flat_obs, update=True)
        
        # Compute multi-objective reward
        total_reward, reward_info = self._compute_multi_objective_reward(
            reward, info, action_list, next_obs
        )
        
        # Update episode stats
        self.episode_reward += total_reward
        self.episode_length += 1
        self.episode_cost += reward_info.get('cost', 0.0)
        self.episode_violations += reward_info.get('violations', 0)
        
        # Add episode stats to info at episode end
        if done or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                'cost': self.episode_cost,
                'violations': self.episode_violations
            }
            self.episode_count += 1
        
        # Add step-level info
        info.update(reward_info)
        
        return flat_obs, total_reward, done, truncated, info
    
    def _flatten_obs(self, obs: List[np.ndarray]) -> np.ndarray:
        """
        Flatten list of observations into single array.
        
        Args:
            obs: List of observations (one per building)
            
        Returns:
            Flattened observation array
        """
        return np.concatenate([np.array(o).flatten() for o in obs])
    
    def _split_action(self, action: np.ndarray) -> List[np.ndarray]:
        """
        Split flattened action into per-building actions.
        
        Args:
            action: Flattened action array
            
        Returns:
            List of per-building actions
        """
        actions = []
        for i in range(self.num_buildings):
            start_idx = i * self.action_dim_per_building
            end_idx = start_idx + self.action_dim_per_building
            actions.append(action[start_idx:end_idx])
        return actions

    def filter_actions(self, flat_action: np.ndarray, obs: np.ndarray = None) -> (np.ndarray, Dict):
        """
        Filter / project a flattened action into the safety constraints.

        Returns filtered flattened action and info about any changes made.
        """
        info = {'filtered': False, 'scale': 1.0, 'reason': ''}

        # Split action per building
        actions = self._split_action(flat_action)

        max_power = None
        if self.safety_constraints is not None:
            max_power = self.safety_constraints.max_power

        if max_power is None:
            # Nothing to enforce
            return flat_action, info

        # Compute per-building magnitudes and scale if necessary
        scales = []
        for act in actions:
            mag = float(np.abs(np.array(act)).sum())
            if mag > max_power:
                scales.append(max_power / (mag + 1e-8))
            else:
                scales.append(1.0)

        # If any scaling needed, apply the minimum scale across buildings
        if any(s < 1.0 for s in scales):
            overall_scale = min(scales)
            filtered = (flat_action * overall_scale).astype(np.float32)
            info['filtered'] = True
            info['scale'] = float(overall_scale)
            info['reason'] = 'max_power_clipping'
            return filtered, info

        return flat_action, info
    
    def _compute_multi_objective_reward(
        self,
        baseline_rewards,
        info: Dict,
        actions: List[np.ndarray],
        next_obs: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Compute weighted multi-objective reward using TRUE CityLearn metrics.
        
        NOW EXTRACTS ACTUAL METRICS from observations:
        - Cost: electricity_price × net_consumption
        - Carbon: carbon_intensity × net_consumption  
        - Peak: maximum instantaneous power
        - Health: battery SoC extremity + discharge rate
        
        Args:
            baseline_rewards: Rewards from CityLearn (list of per-building rewards or scalar)
            info: Info dict from environment
            actions: Actions taken per building
            next_obs: Next observations per building
            
        Returns:
            (scalar_reward, reward_info_dict)
        """
        reward_info = {}
        
        # Extract TRUE metrics from CityLearn observations
        try:
            metrics = self.metric_extractor.extract_metrics(next_obs, actions)
            
            # Store raw metrics for logging
            reward_info['raw_cost'] = metrics['cost']
            reward_info['raw_carbon'] = metrics['carbon']
            reward_info['raw_peak'] = metrics['peak']
            reward_info['raw_battery_soc'] = metrics['battery_soc']
            reward_info['raw_battery_discharge'] = metrics['battery_discharge_rate']
            
            # Compute multi-objective reward with proper weights
            weights = {
                'cost': self.weight_cost,
                'carbon': self.weight_carbon,
                'peak': self.weight_peak,
                'health': self.weight_health
            }
            
            total_reward, components = self.metric_extractor.compute_multi_objective_reward(
                metrics,
                weights,
                baseline_cost=100.0,  # These should be tuned based on environment
                baseline_carbon=10.0,
                baseline_peak=5.0
            )
            
            # Store reward components
            reward_info['cost_component'] = components['cost']
            reward_info['carbon_component'] = components['carbon']
            reward_info['peak_component'] = components['peak']
            reward_info['health_component'] = components['health']
            
        except Exception as e:
            # Fallback to proxy-based if extraction fails
            print(f"Warning: Metric extraction failed: {e}. Using proxy-based reward.")
            
            # Convert baseline rewards to scalar
            if isinstance(baseline_rewards, (list, tuple, np.ndarray)):
                base_reward = float(np.sum(baseline_rewards))
            else:
                base_reward = float(baseline_rewards)
            
            # Use simple proxy
            cost_component = np.clip(base_reward / 1000.0, -1.0, 0.1)
            action_mag = np.mean([np.abs(a).sum() for a in actions])
            normalized_action_mag = np.clip(action_mag / 10.0, 0.0, 1.0)
            
            carbon_component = -0.2 * normalized_action_mag
            peak_component = -0.1 * normalized_action_mag
            health_component = -0.15 * normalized_action_mag
            
            total_reward = (
                self.weight_cost * cost_component +
                self.weight_carbon * carbon_component +
                self.weight_peak * peak_component +
                self.weight_health * health_component
            )
            
            reward_info['cost_component'] = cost_component
            reward_info['carbon_component'] = carbon_component
            reward_info['peak_component'] = peak_component
            reward_info['health_component'] = health_component
        
        # ===== SAFETY CONSTRAINTS =====
        constraint_penalty = 0.0
        violations = 0
        
        if self.safety_constraints is not None:
            # Extract battery SOC from observations for safety checking
            try:
                for obs_idx, obs in enumerate(next_obs):
                    obs_array = np.array(obs).flatten()
                    
                    # Try to extract actual battery SOC
                    if len(obs_array) > self.metric_extractor.BATTERY_SOC_IDX:
                        battery_soc = obs_array[self.metric_extractor.BATTERY_SOC_IDX]
                        
                        # Check SOC constraints
                        if battery_soc < self.safety_constraints.soc_min:
                            constraint_penalty -= 50.0  # Large penalty for low SOC
                            violations += 1
                        elif battery_soc > self.safety_constraints.soc_max:
                            constraint_penalty -= 50.0  # Large penalty for high SOC
                            violations += 1
                    
                    # Check for extreme normalized observations (backup check)
                    extreme_obs = np.sum(np.abs(obs_array) > 3.0)
                    if extreme_obs > 0:
                        constraint_penalty -= 0.5 * extreme_obs
                        violations += max(0, int(extreme_obs) - 1)  # Don't double count
                        
            except Exception as e:
                print(f"Warning: Safety constraint checking failed: {e}")
        
        reward_info['constraint_penalty'] = constraint_penalty
        reward_info['violations'] = violations
        
        # ===== FINAL REWARD =====
        final_reward = total_reward + constraint_penalty
        
        # Optional reward normalization/clipping
        if self.normalize_reward:
            final_reward = np.clip(final_reward, -self.clip_reward, self.clip_reward)
        
        reward_info['final_reward'] = float(final_reward)
        reward_info['cost'] = reward_info.get('raw_cost', 0.0)  # For episode tracking
        
        return float(final_reward), reward_info
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Get metrics from current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'episode_return': self.episode_reward,
            'episode_length': self.episode_length,
            'episode_cost': self.episode_cost,
            'constraint_violations': self.episode_violations,
            'avg_reward_per_step': self.episode_reward / max(1, self.episode_length)
        }
