"""
Seeding and Reproducibility Utilities

Ensures deterministic behavior across all random number generators.
Based on best practices from RL research community.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Make PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng(seed: int = None) -> np.random.Generator:
    """
    Get a numpy random number generator.
    
    Args:
        seed: Random seed (optional)
        
    Returns:
        numpy random generator
    """
    return np.random.default_rng(seed)
