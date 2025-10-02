"""
Curriculum Learning Module for Therapeutic Conversation Training

Implements sophisticated curriculum learning strategies to gradually mix
CounselChat and AnnoMI datasets for optimal learning progression.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from utils import Logger
from data_preprocessing import Conversation, DatasetManager


@dataclass
class CurriculumPhase:
    """Represents a single phase in curriculum learning."""
    phase_id: int
    name: str
    counselchat_ratio: float
    annomi_ratio: float
    epochs: int
    learning_rate: float
    description: str
    
    def __post_init__(self):
        """Validate phase configuration."""
        if abs(self.counselchat_ratio + self.annomi_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {self.counselchat_ratio + self.annomi_ratio}")
        
        if self.counselchat_ratio < 0 or self.annomi_ratio < 0:
            raise ValueError("Ratios must be non-negative")
        
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


class BaseCurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or Logger("CurriculumStrategy").get_logger()
    
    @abstractmethod
    def create_phases(self, **kwargs) -> List[CurriculumPhase]:
        """Create curriculum phases."""
        pass
    
    @abstractmethod
    def should_advance_phase(self, current_phase: CurriculumPhase, metrics: Dict[str, float]) -> bool:
        """Determine if we should advance to the next phase."""
        pass


class TherapeuticCurriculumStrategy(BaseCurriculumStrategy):
    """
    Curriculum strategy specifically designed for therapeutic conversation training.
    
    Gradually transitions from general counseling (CounselChat) to specialized 
    motivational interviewing (AnnoMI) techniques.
    """
    
    def create_phases(self, **kwargs) -> List[CurriculumPhase]:
        """Create therapeutic curriculum phases."""
        
        phases = [
            # Phase 1: Foundation - Pure CounselChat
            CurriculumPhase(
                phase_id=1,
                name="Foundation",
                counselchat_ratio=1.0,
                annomi_ratio=0.0,
                epochs=2,
                learning_rate=5e-5,
                description="Learn basic therapeutic conversation patterns from CounselChat"
            ),
            
            # Phase 2: Introduction - Mostly CounselChat with some AnnoMI
            CurriculumPhase(
                phase_id=2,
                name="Introduction",
                counselchat_ratio=0.8,
                annomi_ratio=0.2,
                epochs=2,
                learning_rate=4e-5,
                description="Introduce AnnoMI techniques while maintaining CounselChat foundation"
            ),
            
            # Phase 3: Integration - Balanced mix
            CurriculumPhase(
                phase_id=3,
                name="Integration",
                counselchat_ratio=0.6,
                annomi_ratio=0.4,
                epochs=3,
                learning_rate=3e-5,
                description="Balance general counseling with motivational interviewing"
            ),
            
            # Phase 4: Specialization - AnnoMI focus
            CurriculumPhase(
                phase_id=4,
                name="Specialization",
                counselchat_ratio=0.3,
                annomi_ratio=0.7,
                epochs=3,
                learning_rate=2e-5,
                description="Focus on advanced motivational interviewing techniques"
            ),
            
            # Phase 5: Mastery - Pure AnnoMI
            CurriculumPhase(
                phase_id=5,
                name="Mastery",
                counselchat_ratio=0.0,
                annomi_ratio=1.0,
                epochs=2,
                learning_rate=1e-5,
                description="Master deep motivational interviewing conversations"
            )
        ]
        
        self.logger.info(f"Created {len(phases)} curriculum phases")
        for phase in phases:
            self.logger.info(f"Phase {phase.phase_id}: {phase.name} - "
                           f"CC:{phase.counselchat_ratio:.1%}, MI:{phase.annomi_ratio:.1%}")
        
        return phases
    
    def should_advance_phase(self, current_phase: CurriculumPhase, metrics: Dict[str, float]) -> bool:
        """
        Determine if we should advance based on loss convergence.
        
        Simple strategy: advance if eval_loss is reasonable or we've completed the epochs.
        """
        eval_loss = metrics.get('eval_loss', float('inf'))
        
        # Define loss thresholds for each phase
        loss_thresholds = {
            1: 2.5,  # Foundation phase - allow higher loss
            2: 2.3,  # Introduction phase
            3: 2.1,  # Integration phase
            4: 1.9,  # Specialization phase
            5: 1.8   # Mastery phase - lowest loss
        }
        
        threshold = loss_thresholds.get(current_phase.phase_id, 2.0)
        should_advance = eval_loss <= threshold
        
        self.logger.info(f"Phase {current_phase.phase_id} - Loss: {eval_loss:.3f}, "
                        f"Threshold: {threshold:.3f}, Advance: {should_advance}")
        
        return should_advance


class AdaptiveCurriculumStrategy(BaseCurriculumStrategy):
    """
    Adaptive curriculum strategy that adjusts based on performance metrics.
    """
    
    def __init__(self, adaptation_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_threshold = adaptation_threshold
        self.performance_history = []
    
    def create_phases(self, **kwargs) -> List[CurriculumPhase]:
        """Create adaptive curriculum phases."""
        
        # Start with a base curriculum similar to therapeutic
        base_phases = TherapeuticCurriculumStrategy().create_phases()
        
        # Add adaptive elements
        for phase in base_phases:
            # Reduce epochs for faster adaptation
            phase.epochs = max(1, phase.epochs - 1)
            # Slightly higher learning rates for adaptation
            phase.learning_rate *= 1.2
        
        return base_phases
    
    def should_advance_phase(self, current_phase: CurriculumPhase, metrics: Dict[str, float]) -> bool:
        """Adaptive advancement based on performance trends."""
        
        eval_loss = metrics.get('eval_loss', float('inf'))
        self.performance_history.append(eval_loss)
        
        # Need at least 2 data points to check trend
        if len(self.performance_history) < 2:
            return False
        
        # Check if loss is improving
        recent_losses = self.performance_history[-3:]  # Last 3 measurements
        if len(recent_losses) >= 2:
            improvement = recent_losses[0] - recent_losses[-1]
            improvement_rate = improvement / recent_losses[0] if recent_losses[0] > 0 else 0
            
            # Advance if improvement is less than threshold (converged)
            should_advance = improvement_rate < self.adaptation_threshold
            
            self.logger.info(f"Adaptive Phase {current_phase.phase_id} - "
                           f"Improvement rate: {improvement_rate:.3f}, "
                           f"Threshold: {self.adaptation_threshold:.3f}, "
                           f"Advance: {should_advance}")
            
            return should_advance
        
        return False


class CurriculumDatasetMixer:
    """Handles mixing of datasets according to curriculum phases."""
    
    def __init__(self, 
                 counselchat_conversations: List[Conversation],
                 annomi_conversations: List[Conversation],
                 dataset_manager: DatasetManager,
                 logger: Optional[logging.Logger] = None):
        
        self.counselchat_conversations = counselchat_conversations
        self.annomi_conversations = annomi_conversations
        self.dataset_manager = dataset_manager
        self.logger = logger or Logger("CurriculumMixer").get_logger()
        
        self.logger.info(f"Initialized mixer with {len(counselchat_conversations)} CounselChat "
                        f"and {len(annomi_conversations)} AnnoMI conversations")
    
    def create_phase_dataset(self, 
                           phase: CurriculumPhase, 
                           target_size: int = 2000,
                           validation_split: float = 0.1) -> Dict[str, Dataset]:
        """Create mixed dataset for a specific curriculum phase."""
        
        self.logger.info(f"Creating dataset for Phase {phase.phase_id}: {phase.name}")
        
        # Calculate dataset sizes
        counselchat_size = int(target_size * phase.counselchat_ratio)
        annomi_size = int(target_size * phase.annomi_ratio)
        
        # Adjust for rounding
        total_calculated = counselchat_size + annomi_size
        if total_calculated < target_size:
            # Add remainder to the larger dataset
            if phase.counselchat_ratio >= phase.annomi_ratio:
                counselchat_size += target_size - total_calculated
            else:
                annomi_size += target_size - total_calculated
        
        self.logger.info(f"Target sizes - CounselChat: {counselchat_size}, AnnoMI: {annomi_size}")
        
        # Sample conversations
        sampled_conversations = []
        
        if counselchat_size > 0:
            cc_sample = self._sample_conversations(
                self.counselchat_conversations, 
                counselchat_size, 
                "CounselChat"
            )
            sampled_conversations.extend(cc_sample)
        
        if annomi_size > 0:
            mi_sample = self._sample_conversations(
                self.annomi_conversations, 
                annomi_size, 
                "AnnoMI"
            )
            sampled_conversations.extend(mi_sample)
        
        # Shuffle the combined dataset
        random.shuffle(sampled_conversations)
        
        # Convert to SFT format
        sft_data = []
        for conv in sampled_conversations:
            formatted = self.dataset_manager.format_for_sft(conv)
            if formatted:
                sft_data.append(formatted)
        
        # Create train/validation split
        val_size = int(len(sft_data) * validation_split)
        train_data = sft_data[val_size:]
        val_data = sft_data[:val_size]
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None
        
        self.logger.info(f"Created datasets - Train: {len(train_dataset)}, "
                        f"Validation: {len(val_dataset) if val_dataset else 0}")
        
        result = {'train': train_dataset}
        if val_dataset:
            result['validation'] = val_dataset
        
        return result
    
    def _sample_conversations(self, 
                            conversations: List[Conversation], 
                            target_size: int, 
                            dataset_name: str) -> List[Conversation]:
        """Sample conversations from a dataset."""
        
        if len(conversations) <= target_size:
            self.logger.info(f"Using all {len(conversations)} {dataset_name} conversations")
            return conversations.copy()
        
        # Sample without replacement
        sampled = random.sample(conversations, target_size)
        self.logger.info(f"Sampled {len(sampled)} from {len(conversations)} {dataset_name} conversations")
        
        return sampled


class CurriculumTrainingManager:
    """Manages the complete curriculum learning process."""
    
    def __init__(self,
                 strategy: BaseCurriculumStrategy,
                 dataset_mixer: CurriculumDatasetMixer,
                 logger: Optional[logging.Logger] = None):
        
        self.strategy = strategy
        self.dataset_mixer = dataset_mixer
        self.logger = logger or Logger("CurriculumManager").get_logger()
        
        self.phases = self.strategy.create_phases()
        self.current_phase_idx = 0
        self.phase_results = {}
        
        self.logger.info(f"Initialized curriculum with {len(self.phases)} phases")
    
    def get_current_phase(self) -> Optional[CurriculumPhase]:
        """Get the current curriculum phase."""
        if self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        return None
    
    def create_current_phase_dataset(self, target_size: int = 2000) -> Dict[str, Dataset]:
        """Create dataset for the current phase."""
        current_phase = self.get_current_phase()
        if current_phase is None:
            raise RuntimeError("No current phase available")
        
        return self.dataset_mixer.create_phase_dataset(current_phase, target_size)
    
    def advance_phase(self, metrics: Dict[str, float]) -> bool:
        """
        Try to advance to the next phase based on current metrics.
        
        Returns:
            bool: True if advanced to next phase, False if staying in current phase
        """
        current_phase = self.get_current_phase()
        if current_phase is None:
            return False
        
        should_advance = self.strategy.should_advance_phase(current_phase, metrics)
        
        if should_advance and self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            next_phase = self.get_current_phase()
            self.logger.info(f"Advanced to Phase {next_phase.phase_id}: {next_phase.name}")
            return True
        elif should_advance:
            self.logger.info("Curriculum complete - all phases finished")
            return False
        else:
            self.logger.info(f"Staying in Phase {current_phase.phase_id}: {current_phase.name}")
            return False
    
    def record_phase_results(self, phase_id: int, results: Dict[str, Any]):
        """Record results for a completed phase."""
        self.phase_results[phase_id] = results
        self.logger.info(f"Recorded results for Phase {phase_id}")
    
    def is_curriculum_complete(self) -> bool:
        """Check if the curriculum is complete."""
        return self.current_phase_idx >= len(self.phases)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the curriculum training."""
        return {
            'total_phases': len(self.phases),
            'completed_phases': len(self.phase_results),
            'current_phase_idx': self.current_phase_idx,
            'phase_results': self.phase_results,
            'phases_config': [
                {
                    'phase_id': p.phase_id,
                    'name': p.name,
                    'counselchat_ratio': p.counselchat_ratio,
                    'annomi_ratio': p.annomi_ratio,
                    'epochs': p.epochs,
                    'learning_rate': p.learning_rate
                }
                for p in self.phases
            ]
        }


def create_therapeutic_curriculum(counselchat_conversations: List[Conversation],
                                annomi_conversations: List[Conversation],
                                dataset_manager: DatasetManager,
                                strategy_type: str = "therapeutic",
                                logger: Optional[logging.Logger] = None) -> CurriculumTrainingManager:
    """
    Create a curriculum training manager for therapeutic conversation training.
    
    Args:
        counselchat_conversations: List of CounselChat conversations
        annomi_conversations: List of AnnoMI conversations  
        dataset_manager: DatasetManager instance
        strategy_type: Type of curriculum strategy ("therapeutic" or "adaptive")
        logger: Optional logger instance
        
    Returns:
        CurriculumTrainingManager: Configured curriculum manager
    """
    
    if logger is None:
        logger = Logger("CurriculumFactory").get_logger()
    
    # Create strategy
    if strategy_type == "therapeutic":
        strategy = TherapeuticCurriculumStrategy(logger=logger)
    elif strategy_type == "adaptive":
        strategy = AdaptiveCurriculumStrategy(logger=logger)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Create dataset mixer
    mixer = CurriculumDatasetMixer(
        counselchat_conversations=counselchat_conversations,
        annomi_conversations=annomi_conversations,
        dataset_manager=dataset_manager,
        logger=logger
    )
    
    # Create manager
    manager = CurriculumTrainingManager(
        strategy=strategy,
        dataset_mixer=mixer,
        logger=logger
    )
    
    logger.info(f"Created {strategy_type} curriculum with {len(manager.phases)} phases")
    
    return manager 