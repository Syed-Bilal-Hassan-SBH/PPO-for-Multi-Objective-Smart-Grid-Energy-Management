"""
Configuration Loading Utilities

Handles loading and merging YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration object with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)
    
    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration object with attribute access
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
