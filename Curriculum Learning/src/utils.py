"""
Utility functions for Gemma 3n fine-tuning pipeline.
Provides configuration loading, logging, model utilities, and helper functions.
"""

import os
import json
import yaml
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import wandb
from transformers import set_seed
import psutil


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get loaded configuration or load if not cached."""
        if config_name not in self.configs:
            return self.load_config(config_name)
        return self.configs[config_name]
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """Save configuration to YAML file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


class Logger:
    """Enhanced logging utility with file and console output."""
    
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        return self.logger


class ModelUtils:
    """Utilities for model operations and memory management."""
    
    @staticmethod
    def get_model_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            memory_info = {}
            for i in range(torch.cuda.device_count()):
                memory_info[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    'cached': torch.cuda.memory_reserved(i) / 1024**3,      # GB
                    'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                }
            return memory_info
        return {}
    
    @staticmethod
    def get_system_memory_usage() -> Dict[str, float]:
        """Get system memory usage."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024**3,      # GB
            'available': memory.available / 1024**3,  # GB
            'used': memory.used / 1024**3,        # GB
            'percentage': memory.percent
        }
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def count_parameters(model) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    @staticmethod
    def print_model_info(model, logger: logging.Logger):
        """Print comprehensive model information."""
        param_counts = ModelUtils.count_parameters(model)
        memory_usage = ModelUtils.get_model_memory_usage()
        
        logger.info("=" * 50)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Total parameters: {param_counts['total']:,}")
        logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
        logger.info(f"Frozen parameters: {param_counts['frozen']:,}")
        logger.info(f"Trainable percentage: {param_counts['trainable']/param_counts['total']*100:.2f}%")
        
        if memory_usage:
            logger.info("\nGPU Memory Usage:")
            for gpu, info in memory_usage.items():
                logger.info(f"  {gpu}: {info['allocated']:.2f}GB allocated, {info['cached']:.2f}GB cached")
        
        system_memory = ModelUtils.get_system_memory_usage()
        logger.info(f"\nSystem Memory: {system_memory['used']:.2f}GB / {system_memory['total']:.2f}GB ({system_memory['percentage']:.1f}%)")
        logger.info("=" * 50)


class ExperimentTracker:
    """Experiment tracking and logging utilities."""
    
    def __init__(self, project_name: str, experiment_name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None):
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.tags = tags or []
        
        # Initialize wandb if available
        try:
            wandb.init(
                project=project_name,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                reinit=True
            )
            self.use_wandb = True
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to tracking system."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact to tracking system."""
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_dir(artifact_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish experiment tracking."""
        if self.use_wandb:
            wandb.finish()


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directory(base_dir: str, experiment_name: str) -> Path:
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_training_args(args, output_dir: Union[str, Path]):
    """Save training arguments to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert args to dict if it's not already
    if hasattr(args, 'to_dict'):
        args_dict = args.to_dict()
    elif hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    # Save to JSON
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)


def load_training_args(output_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load training arguments from JSON file."""
    args_path = Path(output_dir) / 'training_args.json'
    
    if not args_path.exists():
        raise FileNotFoundError(f"Training args file not found: {args_path}")
    
    with open(args_path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        device_info['gpu_memory'] = [torch.cuda.get_device_properties(i).total_memory / 1024**3 
                                   for i in range(torch.cuda.device_count())]
    
    return device_info


class ProgressTracker:
    """Track training progress and provide estimates."""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.current_step = 0
        self.total_steps = 0
    
    def start(self, total_steps: int):
        """Start tracking progress."""
        self.start_time = datetime.now()
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times = []
    
    def update(self, step: int):
        """Update current step."""
        if self.start_time is None:
            return
        
        current_time = datetime.now()
        if self.current_step > 0:
            step_time = (current_time - self.last_update_time).total_seconds()
            self.step_times.append(step_time)
            
            # Keep only last 100 step times for moving average
            if len(self.step_times) > 100:
                self.step_times = self.step_times[-100:]
        
        self.current_step = step
        self.last_update_time = current_time
    
    def get_eta(self) -> Optional[str]:
        """Get estimated time of arrival."""
        if not self.step_times or self.current_step == 0:
            return None
        
        avg_step_time = np.mean(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_step_time
        
        return format_time(eta_seconds)
    
    def get_elapsed_time(self) -> Optional[str]:
        """Get elapsed time since start."""
        if self.start_time is None:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return format_time(elapsed)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information."""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percentage': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            'elapsed_time': self.get_elapsed_time(),
            'eta': self.get_eta(),
            'steps_per_second': 1 / np.mean(self.step_times) if self.step_times else None
        }


# Global instances for easy access
config_manager = ConfigManager()
