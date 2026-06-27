"""
Logging Infrastructure

Provides comprehensive logging for training, including:
- TensorBoard integration
- File logging
- Console output
- Metric tracking
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Comprehensive logging system for RL experiments.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        verbose: int = 1
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            verbose: Verbosity level (0: silent, 1: info, 2: debug)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.verbose = verbose
        
        # Create log directory
        self.experiment_dir = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        self._setup_file_logging()
        
        # Set up TensorBoard
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))
        
        # Metric storage
        self.metrics = {}
        self.step = 0
        
    def _setup_file_logging(self):
        """Set up file logging."""
        log_file = self.experiment_dir / "training.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if self.verbose >= 2 else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric
            value: Value to log
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
            
        # Store metric
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append((step, value))
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, tag: str, values: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple related scalars.
        
        Args:
            tag: Group name
            values: Dictionary of metric names to values
            step: Step number
        """
        if step is None:
            step = self.step
            
        if self.writer is not None:
            self.writer.add_scalars(tag, values, step)

    def log_dict(self, stats: Dict[str, float], step: Optional[int] = None):
        """Convenience for logging a dictionary of scalars under the default 'train' group.

        Args:
            stats: dict of scalar values
            step: step number
        """
        if step is None:
            step = self.step

        # Store each metric and write to TensorBoard
        for k, v in stats.items():
            try:
                self.log_scalar(f"train/{k}", float(v), step)
            except Exception:
                # Fallback: store as string if not floatable
                self.log_text(f"train/{k}", str(v), step)
            
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log a histogram.
        
        Args:
            tag: Name of the histogram
            values: Array of values
            step: Step number
        """
        if step is None:
            step = self.step
            
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
            
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        if self.verbose >= 1:
            self.logger.info(f"Configuration saved to {config_file}")
            
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text.
        
        Args:
            tag: Tag for the text
            text: Text to log
            step: Step number
        """
        if step is None:
            step = self.step
            
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
            
    def info(self, message: str):
        """Log info message."""
        if self.verbose >= 1:
            self.logger.info(message)
            
    def debug(self, message: str):
        """Log debug message."""
        if self.verbose >= 2:
            self.logger.debug(message)
            
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def increment_step(self):
        """Increment internal step counter."""
        self.step += 1
        
    def save_metrics(self):
        """Save all metrics to file."""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert to serializable format
            serializable_metrics = {
                tag: [(int(step), float(value)) for step, value in values]
                for tag, values in self.metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)
            
        if self.verbose >= 1:
            self.logger.info(f"Metrics saved to {metrics_file}")
            
    def close(self):
        """Close logger and save all data."""
        self.save_metrics()
        if self.writer is not None:
            self.writer.close()
