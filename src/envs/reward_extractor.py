"""
CityLearn Metric Extractor

Extracts true metrics from CityLearn observations for multi-objective reward computation.
Based on CityLearn observation structure as documented in official CityLearn docs.
"""

import numpy as np
from typing import Dict, List, Tuple


class CityLearnMetricExtractor:
    """
    Extracts cost, carbon, peak demand, and battery health metrics from CityLearn observations.
    
    CityLearn Observation Structure (per building):
    - Hour of day (normalized)
    - Day type (weekday/weekend)
    - Month
    - Outdoor temperature
    - Solar generation
    - Non-shiftable load
    - DHW (Domestic Hot Water) demand  
    - Cooling demand
    - Heating demand
    - Electricity pricing
    - Carbon intensity
    - Battery SOC (State of Charge)
    - Net electricity consumption
    """
    
    def __init__(self, num_buildings: int = 5):
        """
        Initialize metric extractor.
        
        Args:
            num_buildings: Number of buildings in environment
        """
        self.num_buildings = num_buildings
        
        # CityLearn observation indices (approximate - may vary by schema)
        # These are typical indices, should be verified against actual environment
        self.HOUR_IDX = 0
        self.DAY_TYPE_IDX = 1
        self.MONTH_IDX = 2
        self.OUTDOOR_TEMP_IDX = 3
        self.SOLAR_GEN_IDX = 4
        self.NON_SHIFTABLE_LOAD_IDX = 5
        self.DHW_DEMAND_IDX = 6
        self.COOLING_DEMAND_IDX = 7
        self.HEATING_DEMAND_IDX = 8
        self.ELECTRICITY_PRICE_IDX = 9
        self.CARBON_INTENSITY_IDX = 10
        self.BATTERY_SOC_IDX = 11
        self.NET_CONSUMPTION_IDX = 12
        
        # Running statistics for normalization
        self.cost_mean = 0.0
        self.cost_std = 1.0
        self.carbon_mean = 0.0
        self.carbon_std = 1.0
        self.peak_mean = 0.0
        self.peak_std = 1.0
        self.n_updates = 0
        
    def extract_metrics(
        self, 
        observations: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract all metrics from raw CityLearn observations.
        
        Args:
            observations: List of observations (one per building)
            actions: List of actions taken (one per building)
            
        Returns:
            Dictionary with extracted metrics:
                - cost: Total electricity cost
                - carbon: Total carbon emissions
                -peak: Peak net demand
                - battery_soc: Average battery state of charge
                - battery_discharge_rate: Average discharge rate
        """
        metrics = {}
        
        # Extract per-building metrics
        electricity_prices = []
        carbon_intensities = []
        net_consumptions = []
        battery_socs = []
        battery_actions = []
        
        for i, obs in enumerate(observations):
            obs_array = np.array(obs).flatten()
            
            # Handle different observation lengths gracefully
            if len(obs_array) > self.ELECTRICITY_PRICE_IDX:
                electricity_prices.append(obs_array[self.ELECTRICITY_PRICE_IDX])
            else:
                electricity_prices.append(1.0)  # Default price
                
            if len(obs_array) > self.CARBON_INTENSITY_IDX:
                carbon_intensities.append(obs_array[self.CARBON_INTENSITY_IDX])
            else:
                carbon_intensities.append(0.5)  # Default carbon intensity
                
            if len(obs_array) > self.NET_CONSUMPTION_IDX:
                net_consumptions.append(obs_array[self.NET_CONSUMPTION_IDX])
            else:
                # Estimate from actions if not available
                net_consumptions.append(np.sum(np.abs(actions[i])))
                
            if len(obs_array) > self.BATTERY_SOC_IDX:
                battery_socs.append(obs_array[self.BATTERY_SOC_IDX])
            else:
                battery_socs.append(0.5)  # Default middle SOC
                
            battery_actions.append(actions[i])
        
        # Aggregate metrics across buildings
        electricity_prices = np.array(electricity_prices)
        carbon_intensities = np.array(carbon_intensities)
        net_consumptions = np.array(net_consumptions)
        battery_socs = np.array(battery_socs)
        
        # COST: sum of (net_consumption * price) for all buildings
        total_cost = np.sum(net_consumptions * electricity_prices)
        metrics['cost'] = float(total_cost)
        
        # CARBON: sum of (net_consumption * carbon_intensity) for all buildings  
        total_carbon = np.sum(net_consumptions * carbon_intensities)
        metrics['carbon'] = float(total_carbon)
        
        # PEAK DEMAND: maximum instantaneous consumption across all buildings
        peak_demand = np.max(np.abs(net_consumptions))
        metrics['peak'] = float(peak_demand)
        
        # BATTERY HEALTH metrics
        avg_soc = np.mean(battery_socs)
        metrics['battery_soc'] = float(avg_soc)
        
        # Discharge rate (negative actions mean discharge)
        battery_actions_array = np.array([np.array(a).flatten() for a in battery_actions])
        discharge_amounts = np.minimum(battery_actions_array, 0)  # Only negative (discharge)
        avg_discharge = np.abs(np.mean(discharge_amounts))
        metrics['battery_discharge_rate'] = float(avg_discharge)
        
        # Extreme SOC penalty (health degradation from extreme states)
        soc_extremity = np.mean(np.abs(battery_socs - 0.5) * 2)  # 0 at 50%, 1 at 0% or 100%
        metrics['battery_extremity'] = float(soc_extremity)
        
        return metrics
    
    def compute_multi_objective_reward(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
        baseline_cost: float = 100.0,
        baseline_carbon: float = 10.0,
        baseline_peak: float = 5.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted multi-objective reward from extracted metrics.
        
        Args:
            metrics: Dictionary of extracted metrics
            weights: Dictionary of weights for each objective
            baseline_cost: Baseline cost for normalization
            baseline_carbon: Baseline carbon for normalization
            baseline_peak: Baseline peak for normalization
            
        Returns:
            (total_reward, component_rewards_dict)
        """
        component_rewards = {}
        
        # ===== COST COMPONENT =====
        # Minimize cost - reward is negative cost, normalized
        cost_normalized = -metrics['cost'] / max(baseline_cost, 1.0)
        cost_component = cost_normalized
        component_rewards['cost'] = float(cost_component)
        
        # ===== CARBON COMPONENT =====
        # Minimize carbon emissions
        carbon_normalized = -metrics['carbon'] / max(baseline_carbon, 1.0)
        carbon_component = carbon_normalized
        component_rewards['carbon'] = float(carbon_component)
        
        # ===== PEAK DEMAND COMPONENT =====
        # Minimize peak demand (grid stability)
        peak_normalized = -metrics['peak'] / max(baseline_peak, 1.0)
        peak_component = peak_normalized
        component_rewards['peak'] = float(peak_component)
        
        # ===== BATTERY HEALTH COMPONENT =====
        # Penalize extreme SOC and high discharge rates
        # Ideal: SOC around 50%, low discharge rates
        soc_penalty = -metrics.get('battery_extremity', 0) * 0.5
        discharge_penalty = -metrics.get('battery_discharge_rate', 0) * 0.3
        health_component = soc_penalty + discharge_penalty
        component_rewards['health'] = float(health_component)
        
        # ===== WEIGHTED SUM =====
        total_reward = (
            weights.get('cost', 0.4) * cost_component +
            weights.get('carbon', 0.2) * carbon_component +
            weights.get('peak', 0.2) * peak_component +
            weights.get('health', 0.2) * health_component
        )
        
        component_rewards['total'] = float(total_reward)
        
        return total_reward, component_rewards


def test_metric_extractor():
    """Test the metric extractor with dummy data."""
    extractor = CityLearnMetricExtractor(num_buildings=2)
    
    # Create dummy observations (simplified)
    obs = [
        np.array([0.5, 0, 6, 20.0, 0.5, 2.0, 1.5, 0.8, 0.3, 0.15, 0.6, 0.5, 2.5]),  # Building 1
        np.array([0.5, 0, 6, 20.0, 0.3, 1.8, 1.2, 0.6, 0.2, 0.15, 0.6, 0.6, 2.2])   # Building 2
    ]
    
    # Create dummy actions
    actions = [
        np.array([0.5]),   # Building 1: slight charge
        np.array([-0.3])   # Building 2: slight discharge
    ]
    
    # Extract metrics
    metrics = extractor.extract_metrics(obs, actions)
    
    print("Extracted Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute reward
    weights = {'cost': 0.4, 'carbon': 0.2, 'peak': 0.2, 'health': 0.2}
    total_reward, components = extractor.compute_multi_objective_reward(metrics, weights)
    
    print("\nReward Components:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
    
    return extractor


if __name__ == "__main__":
    test_metric_extractor()
