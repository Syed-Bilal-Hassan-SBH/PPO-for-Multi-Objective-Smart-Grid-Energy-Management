"""
Metrics Calculation and Tracking

Implements comprehensive metrics for smart grid energy management.
"""

import numpy as np
from typing import Dict, List


def calculate_episode_metrics(episode_data: Dict[str, List]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for an episode.
    
    Args:
        episode_data: Dictionary containing episode data with keys:
            - rewards: List of rewards
            - actions: List of actions
            - observations: List of observations
            
    Returns:
        Dictionary of metrics
    """
    rewards = np.array(episode_data['rewards'])
    
    metrics = {
        # Basic metrics
        'episode_reward': np.sum(rewards),
        'episode_length': len(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
    }
    
    # Action statistics
    if 'actions' in episode_data:
        actions = np.array(episode_data['actions'])
        metrics.update({
            'mean_action': np.mean(actions),
            'std_action': np.std(actions),
            'action_range': np.max(actions) - np.min(actions),
        })
    
    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple episodes.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric keys
    keys = metrics_list[0].keys()
    
    aggregated = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return aggregated


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for printing
    """
    print(f"\n{prefix}Metrics:")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    print("=" * 60)
