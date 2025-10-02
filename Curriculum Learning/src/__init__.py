"""
Gemma 3n Therapeutic Fine-tuning Package

A professional-grade fine-tuning pipeline for Google's Gemma 3n E2B model,
specialized for therapeutic conversations using CounselChat and AnnoMI datasets.

This package provides:
- Data preprocessing for therapeutic conversation datasets
- Multi-stage training pipeline (SFT + DPO)
- Comprehensive evaluation with therapeutic-specific metrics
- Professional utilities and logging
"""

__version__ = "1.0.0"
__author__ = "Gemma 3n Therapeutic Team"
__email__ = "contact@example.com"

# Core modules
from .utils import (
    ConfigManager,
    Logger,
    ModelUtils,
    ExperimentTracker,
    ProgressTracker,
    set_random_seeds,
    create_output_directory,
    get_device_info,
    format_time,
    config_manager
)

from .data_preprocessing import (
    ConversationTurn,
    Conversation,
    BaseDatasetProcessor,
    CounselChatProcessor,
    AnnoMIProcessor,
    DatasetFormatter,
    DatasetManager
)

from .training import (
    BaseTrainer,
    SFTTrainerWrapper,
    DPOTrainerWrapper,
    TrainingPipeline,
    CurriculumTrainingPipeline
)

from .curriculum_learning import (
    CurriculumPhase,
    BaseCurriculumStrategy,
    TherapeuticCurriculumStrategy,
    AdaptiveCurriculumStrategy,
    CurriculumDatasetMixer,
    CurriculumTrainingManager,
    create_therapeutic_curriculum
)

from .eval import (
    EvaluationResult,
    ConversationEvaluation,
    TherapeuticMetrics,
    ModelEvaluator
)

# Package metadata
__all__ = [
    # Utils
    "ConfigManager",
    "Logger", 
    "ModelUtils",
    "ExperimentTracker",
    "ProgressTracker",
    "set_random_seeds",
    "create_output_directory",
    "get_device_info",
    "format_time",
    "config_manager",
    
    # Data preprocessing
    "ConversationTurn",
    "Conversation", 
    "BaseDatasetProcessor",
    "CounselChatProcessor",
    "AnnoMIProcessor",
    "DatasetFormatter",
    "DatasetManager",
    
    # Training
    "BaseTrainer",
    "SFTTrainerWrapper", 
    "DPOTrainerWrapper",
    "TrainingPipeline",
    "CurriculumTrainingPipeline",
    
    # Curriculum Learning
    "CurriculumPhase",
    "BaseCurriculumStrategy",
    "TherapeuticCurriculumStrategy", 
    "AdaptiveCurriculumStrategy",
    "CurriculumDatasetMixer",
    "CurriculumTrainingManager",
    "create_therapeutic_curriculum",
    
    # Evaluation
    "EvaluationResult",
    "ConversationEvaluation",
    "TherapeuticMetrics",
    "ModelEvaluator"
]

# Package configuration
DEFAULT_CONFIG = {
    "model_name": "google/gemma-3n-E2B-it",
    "max_seq_length": 2048,
    "training_stages": ["stage1", "stage2", "dpo"],
    "evaluation_metrics": ["rouge", "bleu", "bert_score", "therapeutic"]
}

def get_version():
    """Get package version."""
    return __version__

def get_default_config():
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()

# Logging setup
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
