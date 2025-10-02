"""
Training module for Gemma 3n fine-tuning.
Supports both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Gemma3nForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training,
    PeftModel, PeftConfig
)
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import Dataset, DatasetDict
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import gc
import os

from utils import (
    Logger, ModelUtils, ExperimentTracker, ProgressTracker,
    save_training_args, format_time, get_device_info
)
from curriculum_learning import CurriculumTrainingManager, create_therapeutic_curriculum


class BaseTrainer:
    """Base class for all trainers with common functionality."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 lora_config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        
        self.model_config = model_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.logger = logger or Logger("BaseTrainer").get_logger()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.experiment_tracker = None
        self.progress_tracker = ProgressTracker()
        
        # Device info
        self.device_info = get_device_info()
        self.logger.info(f"Device info: {self.device_info}")
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """Setup and configure tokenizer."""
        self.logger.info(f"Loading tokenizer: {self.model_config['tokenizer']['name']}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['tokenizer']['name'],
            trust_remote_code=self.model_config.get('trust_remote_code', True),
            padding_side=self.model_config['tokenizer'].get('padding_side', 'right'),
            truncation_side=self.model_config['tokenizer'].get('truncation_side', 'right')
        )
        
        # Set special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Configure tokenizer settings
        tokenizer.add_eos_token = self.model_config['tokenizer'].get('add_eos_token', True)
        tokenizer.add_bos_token = self.model_config['tokenizer'].get('add_bos_token', False)
        
        self.tokenizer = tokenizer
        self.logger.info(f"Tokenizer setup complete. Vocab size: {len(tokenizer)}")
        return tokenizer
    
    def setup_model(self, stage: str = "sft") -> Union[AutoModelForCausalLM, Gemma3nForConditionalGeneration]:
        """Setup and configure model with LoRA."""
        self.logger.info(f"Loading model: {self.model_config['model']['name']}")
        
        # Model loading arguments
        model_kwargs = {
            'pretrained_model_name_or_path': self.model_config['model']['name'],
            'trust_remote_code': self.model_config['model'].get('trust_remote_code', True),
            'torch_dtype': getattr(torch, self.model_config['model'].get('torch_dtype', 'bfloat16')),
            'device_map': self.model_config['model'].get('device_map', 'auto'),
            'attn_implementation': self.model_config['model'].get('attn_implementation', 'flash_attention_2')
        }
        
        # Add quantization if specified
        if self.model_config.get('quantization', {}).get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config['quantization'].get('bnb_4bit_compute_dtype', 'bfloat16')),
                bnb_4bit_quant_type=self.model_config['quantization'].get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=self.model_config['quantization'].get('bnb_4bit_use_double_quant', True)
            )
            model_kwargs['quantization_config'] = bnb_config
        
        # Load model
        try:
            # Try Gemma3n specific model first
            model = Gemma3nForConditionalGeneration.from_pretrained(**model_kwargs)
            self.logger.info("Loaded Gemma3nForConditionalGeneration")
        except Exception as e:
            self.logger.warning(f"Failed to load Gemma3nForConditionalGeneration: {e}")
            # Fallback to standard causal LM
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.logger.info("Loaded AutoModelForCausalLM")
        
        # Prepare model for training
        if self.model_config.get('quantization', {}).get('load_in_4bit', False):
            model = prepare_model_for_kbit_training(model)
        
        # Setup LoRA
        model = self._setup_lora(model, stage)
        
        # Enable gradient checkpointing
        if self.training_config['training'].get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
        
        self.model = model
        
        # Print model information
        ModelUtils.print_model_info(model, self.logger)
        
        return model
    
    def _setup_lora(self, model, stage: str):
        """Setup LoRA configuration."""
        # Get stage-specific LoRA config
        if stage == "stage1" and "stage1_lora" in self.lora_config:
            lora_cfg = self.lora_config["stage1_lora"]
        elif stage == "stage2" and "stage2_lora" in self.lora_config:
            lora_cfg = self.lora_config["stage2_lora"]
        else:
            lora_cfg = self.lora_config["lora"]
        
        self.logger.info(f"Setting up LoRA with config: {lora_cfg}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['lora_alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['lora_dropout'],
            bias=lora_cfg['bias'],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=lora_cfg.get('inference_mode', False)
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        self.logger.info("LoRA setup complete")
        return model
    
    def setup_experiment_tracking(self, experiment_name: str, stage: str):
        """Setup experiment tracking."""
        if self.training_config.get('wandb', {}).get('project'):
            config = {
                'model_config': self.model_config,
                'training_config': self.training_config,
                'lora_config': self.lora_config,
                'stage': stage,
                'device_info': self.device_info
            }
            
            tags = self.training_config['wandb'].get('tags', [])
            tags.append(stage)
            
            self.experiment_tracker = ExperimentTracker(
                project_name=self.training_config['wandb']['project'],
                experiment_name=f"{experiment_name}_{stage}",
                config=config,
                tags=tags
            )
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True):
        """Save model and tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model to: {output_path}")
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_path)
        else:
            torch.save(self.model.state_dict(), output_path / 'pytorch_model.bin')
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save training arguments
        if hasattr(self, 'training_args'):
            save_training_args(self.training_args, output_path)
        
        # Save configs
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        with open(output_path / 'lora_config.json', 'w') as f:
            json.dump(self.lora_config, f, indent=2)
        
        self.logger.info("Model saved successfully")
    
    def load_model(self, model_path: str):
        """Load a previously trained model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        try:
            # Try loading as PEFT model first
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=getattr(torch, self.model_config['model'].get('torch_dtype', 'bfloat16')),
                device_map=self.model_config['model'].get('device_map', 'auto')
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.logger.info("Loaded PEFT model")
        except Exception as e:
            self.logger.warning(f"Failed to load as PEFT model: {e}")
            # Load as regular model
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.logger.info("Loaded regular model")
        
        return self.model, self.tokenizer
    
    def cleanup(self):
        """Cleanup resources."""
        if self.experiment_tracker:
            self.experiment_tracker.finish()
        
        # Clear GPU memory
        ModelUtils.clear_gpu_memory()
        gc.collect()


class SFTTrainerWrapper(BaseTrainer):
    """Wrapper for Supervised Fine-Tuning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = None
    
    def setup_data_collator(self):
        """Setup data collator for SFT."""
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8  # For efficiency
        )
        return self.data_collator
    
    def create_training_arguments(self, output_dir: str, stage: str = "sft") -> TrainingArguments:
        """Create training arguments for SFT."""
        # Get stage-specific config
        if stage == "stage1" and "stage1_training" in self.training_config:
            train_cfg = {**self.training_config["training"], **self.training_config["stage1_training"]}
        elif stage == "stage2" and "stage2_training" in self.training_config:
            train_cfg = {**self.training_config["training"], **self.training_config["stage2_training"]}
        else:
            train_cfg = self.training_config["training"]
        
        # Update output directory
        train_cfg = train_cfg.copy()
        train_cfg['output_dir'] = output_dir
        
        # Create training arguments
        training_args = TrainingArguments(**train_cfg)
        
        self.training_args = training_args
        return training_args
    
    def train(self, 
              train_dataset: Dataset, 
              eval_dataset: Optional[Dataset] = None,
              output_dir: str = "./checkpoints/sft",
              stage: str = "sft",
              experiment_name: str = "gemma3n_sft") -> Dict[str, Any]:
        """Train the model using SFT."""
        
        self.logger.info(f"Starting SFT training - Stage: {stage}")
        
        # Setup components
        if not self.tokenizer:
            self.setup_tokenizer()
        
        if not self.model:
            self.setup_model(stage)
        
        self.setup_data_collator()
        self.setup_experiment_tracking(experiment_name, stage)
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir, stage)
        
        # Setup trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            dataset_text_field="text",  # Field containing the text data
            max_seq_length=self.training_config['training'].get('max_seq_length', 2048),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None
        )
        
        self.trainer = trainer
        
        # Log training info
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Start training
        start_time = time.time()
        self.progress_tracker.start(training_args.max_steps or len(train_dataset) * training_args.num_train_epochs)
        
        try:
            train_result = trainer.train()
            
            # Log results
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {format_time(training_time)}")
            self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Save model
            self.save_model(output_dir)
            
            # Log to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics({
                    'final_train_loss': train_result.training_loss,
                    'training_time_seconds': training_time,
                    'total_steps': train_result.global_step
                })
                self.experiment_tracker.log_artifact(output_dir, "model")
            
            return {
                'training_loss': train_result.training_loss,
                'training_time': training_time,
                'global_step': train_result.global_step,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.cleanup()


class DPOTrainerWrapper(BaseTrainer):
    """Wrapper for Direct Preference Optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = None
    
    def setup_reference_model(self):
        """Setup reference model for DPO."""
        self.logger.info("Setting up reference model for DPO")
        
        # Load reference model (same as main model but frozen)
        model_kwargs = {
            'pretrained_model_name_or_path': self.model_config['model']['name'],
            'trust_remote_code': self.model_config['model'].get('trust_remote_code', True),
            'torch_dtype': getattr(torch, self.model_config['model'].get('torch_dtype', 'bfloat16')),
            'device_map': self.model_config['model'].get('device_map', 'auto')
        }
        
        try:
            self.reference_model = Gemma3nForConditionalGeneration.from_pretrained(**model_kwargs)
        except Exception:
            self.reference_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Reference model setup complete")
        return self.reference_model
    
    def create_dpo_config(self, output_dir: str) -> DPOConfig:
        """Create DPO configuration."""
        dpo_cfg = self.training_config.get('dpo_training', {})
        
        # Merge with base training config
        base_cfg = self.training_config['training'].copy()
        base_cfg.update(dpo_cfg)
        base_cfg['output_dir'] = output_dir
        
        dpo_config = DPOConfig(**base_cfg)
        
        self.training_args = dpo_config
        return dpo_config
    
    def train(self,
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None,
              output_dir: str = "./checkpoints/dpo",
              experiment_name: str = "gemma3n_dpo") -> Dict[str, Any]:
        """Train the model using DPO."""
        
        self.logger.info("Starting DPO training")
        
        # Setup components
        if not self.tokenizer:
            self.setup_tokenizer()
        
        if not self.model:
            self.setup_model("dpo")
        
        self.setup_reference_model()
        self.setup_experiment_tracking(experiment_name, "dpo")
        
        # Create DPO config
        dpo_config = self.create_dpo_config(output_dir)
        
        # Setup DPO trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            beta=dpo_config.beta,
            max_length=dpo_config.max_length,
            max_prompt_length=dpo_config.max_prompt_length
        )
        
        self.trainer = trainer
        
        # Log training info
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Start training
        start_time = time.time()
        
        try:
            train_result = trainer.train()
            
            # Log results
            training_time = time.time() - start_time
            self.logger.info(f"DPO training completed in {format_time(training_time)}")
            self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Save model
            self.save_model(output_dir)
            
            # Log to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics({
                    'final_train_loss': train_result.training_loss,
                    'training_time_seconds': training_time,
                    'total_steps': train_result.global_step
                })
                self.experiment_tracker.log_artifact(output_dir, "model")
            
            return {
                'training_loss': train_result.training_loss,
                'training_time': training_time,
                'global_step': train_result.global_step,
                'output_dir': output_dir
            }
            
        except Exception as e:
            self.logger.error(f"DPO training failed: {e}")
            raise
        finally:
            self.cleanup()


class TrainingPipeline:
    """Complete training pipeline orchestrator."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 lora_config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        
        self.model_config = model_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.logger = logger or Logger("TrainingPipeline").get_logger()
        
        self.results = {}
    
    def run_stage1_sft(self, 
                       train_dataset: Dataset,
                       eval_dataset: Optional[Dataset] = None,
                       output_dir: str = "./checkpoints/stage1") -> Dict[str, Any]:
        """Run Stage 1 SFT (CounselChat)."""
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING STAGE 1: COUNSELCHAT SFT TRAINING")
        self.logger.info("=" * 60)
        
        trainer = SFTTrainerWrapper(
            self.model_config,
            self.training_config,
            self.lora_config,
            self.logger
        )
        
        result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            stage="stage1",
            experiment_name="gemma3n_stage1_counselchat"
        )
        
        self.results['stage1'] = result
        self.logger.info("Stage 1 SFT completed successfully")
        
        return result
    
    def run_stage2_sft(self,
                       train_dataset: Dataset,
                       eval_dataset: Optional[Dataset] = None,
                       stage1_model_path: Optional[str] = None,
                       output_dir: str = "./checkpoints/stage2") -> Dict[str, Any]:
        """Run Stage 2 SFT (AnnoMI)."""
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING STAGE 2: ANNOMI SFT TRAINING")
        self.logger.info("=" * 60)
        
        trainer = SFTTrainerWrapper(
            self.model_config,
            self.training_config,
            self.lora_config,
            self.logger
        )
        
        # Load Stage 1 model if provided
        if stage1_model_path:
            self.logger.info(f"Loading Stage 1 model from: {stage1_model_path}")
            trainer.load_model(stage1_model_path)
        
        result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            stage="stage2",
            experiment_name="gemma3n_stage2_annomi"
        )
        
        self.results['stage2'] = result
        self.logger.info("Stage 2 SFT completed successfully")
        
        return result
    
    def run_dpo(self,
                train_dataset: Dataset,
                eval_dataset: Optional[Dataset] = None,
                sft_model_path: Optional[str] = None,
                output_dir: str = "./checkpoints/dpo") -> Dict[str, Any]:
        """Run DPO training."""
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING DPO TRAINING")
        self.logger.info("=" * 60)
        
        trainer = DPOTrainerWrapper(
            self.model_config,
            self.training_config,
            self.lora_config,
            self.logger
        )
        
        # Load SFT model if provided
        if sft_model_path:
            self.logger.info(f"Loading SFT model from: {sft_model_path}")
            trainer.load_model(sft_model_path)
        
        result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            experiment_name="gemma3n_dpo"
        )
        
        self.results['dpo'] = result
        self.logger.info("DPO training completed successfully")
        
        return result
    
    def run_full_pipeline(self,
                         stage1_datasets: Tuple[Dataset, Optional[Dataset]],
                         stage2_datasets: Tuple[Dataset, Optional[Dataset]],
                         dpo_datasets: Tuple[Dataset, Optional[Dataset]],
                         base_output_dir: str = "./checkpoints") -> Dict[str, Any]:
        """Run the complete training pipeline."""
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE GEMMA 3N FINE-TUNING PIPELINE")
        self.logger.info("=" * 80)
        
        base_path = Path(base_output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: CounselChat SFT
            stage1_result = self.run_stage1_sft(
                train_dataset=stage1_datasets[0],
                eval_dataset=stage1_datasets[1],
                output_dir=str(base_path / "stage1")
            )
            
            # Stage 2: AnnoMI SFT (using Stage 1 model)
            stage2_result = self.run_stage2_sft(
                train_dataset=stage2_datasets[0],
                eval_dataset=stage2_datasets[1],
                stage1_model_path=stage1_result['output_dir'],
                output_dir=str(base_path / "stage2")
            )
            
            # DPO: Preference tuning (using Stage 2 model)
            dpo_result = self.run_dpo(
                train_dataset=dpo_datasets[0],
                eval_dataset=dpo_datasets[1],
                sft_model_path=stage2_result['output_dir'],
                output_dir=str(base_path / "dpo")
            )
            
            # Final results
            pipeline_results = {
                'stage1_sft': stage1_result,
                'stage2_sft': stage2_result,
                'dpo': dpo_result,
                'final_model_path': dpo_result['output_dir'],
                'total_training_time': (
                    stage1_result['training_time'] + 
                    stage2_result['training_time'] + 
                    dpo_result['training_time']
                )
            }
            
            # Save pipeline results
            with open(base_path / 'pipeline_results.json', 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            self.logger.info("=" * 80)
            self.logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
            self.logger.info(f"Final model saved to: {dpo_result['output_dir']}")
            self.logger.info(f"Total training time: {format_time(pipeline_results['total_training_time'])}")
            self.logger.info("=" * 80)
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_results(self) -> Dict[str, Any]:
        """Get all training results."""
        return self.results


class CurriculumTrainingPipeline(TrainingPipeline):
    """Training pipeline with curriculum learning support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_manager = None
    
    def setup_curriculum_learning(self, 
                                counselchat_conversations: List,
                                annomi_conversations: List,
                                dataset_manager,
                                strategy_type: str = "therapeutic"):
        """Setup curriculum learning manager."""
        
        self.logger.info("Setting up curriculum learning...")
        
        self.curriculum_manager = create_therapeutic_curriculum(
            counselchat_conversations=counselchat_conversations,
            annomi_conversations=annomi_conversations,
            dataset_manager=dataset_manager,
            strategy_type=strategy_type,
            logger=self.logger
        )
        
        self.logger.info("Curriculum learning setup complete")
    
    def run_curriculum_training(self, 
                              output_dir: str = "./checkpoints/curriculum",
                              target_dataset_size: int = 2000) -> Dict[str, Any]:
        """Run complete curriculum learning training."""
        
        if self.curriculum_manager is None:
            raise RuntimeError("Curriculum manager not initialized. Call setup_curriculum_learning first.")
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING CURRICULUM LEARNING TRAINING")
        self.logger.info("=" * 80)
        
        base_output_dir = Path(output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        while not self.curriculum_manager.is_curriculum_complete():
            current_phase = self.curriculum_manager.get_current_phase()
            if current_phase is None:
                break
            
            self.logger.info(f"=" * 60)
            self.logger.info(f"PHASE {current_phase.phase_id}: {current_phase.name}")
            self.logger.info(f"CounselChat: {current_phase.counselchat_ratio:.1%}, "
                           f"AnnoMI: {current_phase.annomi_ratio:.1%}")
            self.logger.info(f"=" * 60)
            
            # Create phase dataset
            phase_datasets = self.curriculum_manager.create_current_phase_dataset(target_dataset_size)
            
            # Setup phase-specific training config
            phase_training_config = self.training_config.copy()
            phase_training_config['training']['num_train_epochs'] = current_phase.epochs
            phase_training_config['training']['learning_rate'] = current_phase.learning_rate
            
            # Create phase trainer
            trainer = SFTTrainerWrapper(
                model_config=self.model_config,
                training_config=phase_training_config,
                lora_config=self.lora_config,
                logger=self.logger
            )
            
            # Load previous phase model if available
            if all_results:
                last_phase_id = max(all_results.keys())
                last_model_path = all_results[last_phase_id]['output_dir']
                self.logger.info(f"Loading model from previous phase: {last_model_path}")
                trainer.load_model(last_model_path)
            
            # Train current phase
            phase_output_dir = base_output_dir / f"phase_{current_phase.phase_id}_{current_phase.name.lower()}"
            
            phase_result = trainer.train(
                train_dataset=phase_datasets['train'],
                eval_dataset=phase_datasets['validation'],
                output_dir=str(phase_output_dir),
                stage=f"curriculum_phase_{current_phase.phase_id}",
                experiment_name=f"curriculum_{current_phase.name.lower()}"
            )
            
            # Record phase results
            all_results[current_phase.phase_id] = phase_result
            self.curriculum_manager.record_phase_results(current_phase.phase_id, phase_result)
            
            # Check if we should advance to next phase
            metrics = {'eval_loss': phase_result.get('training_loss', float('inf'))}
            advanced = self.curriculum_manager.advance_phase(metrics)
            
            if not advanced:
                self.logger.info(f"Staying in Phase {current_phase.phase_id} for additional training")
                # Could implement additional training logic here
                break
        
        # Final results
        curriculum_results = {
            'phase_results': all_results,
            'curriculum_summary': self.curriculum_manager.get_training_summary(),
            'final_model_path': all_results[max(all_results.keys())]['output_dir'] if all_results else None,
            'total_phases_completed': len(all_results)
        }
        
        # Save curriculum results
        import json
        with open(base_output_dir / 'curriculum_results.json', 'w') as f:
            json.dump(curriculum_results, f, indent=2, default=str)
        
        self.logger.info("=" * 80)
        self.logger.info("CURRICULUM LEARNING COMPLETED")
        self.logger.info(f"Completed {len(all_results)} phases")
        self.logger.info(f"Final model: {curriculum_results['final_model_path']}")
        self.logger.info("=" * 80)
        
        return curriculum_results
