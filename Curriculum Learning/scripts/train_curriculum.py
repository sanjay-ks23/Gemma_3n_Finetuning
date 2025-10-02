#!/usr/bin/env python3
"""
Curriculum Learning Training Script
Implements sophisticated curriculum learning for Gemma 3n therapeutic fine-tuning.
Gradually mixes CounselChat and AnnoMI datasets for optimal learning progression.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    config_manager, Logger, set_random_seeds, 
    create_output_directory, get_device_info, format_time
)
from data_preprocessing import (
    CounselChatProcessor, AnnoMIProcessor, DatasetManager
)
from training import CurriculumTrainingPipeline
from transformers import AutoTokenizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Curriculum Learning Training for Gemma 3n")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints/curriculum",
        help="Output directory for curriculum training"
    )
    
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./configs",
        help="Directory containing configuration files"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="gemma3n_curriculum_therapeutic",
        help="Name for the curriculum experiment"
    )
    
    parser.add_argument(
        "--counselchat_dataset",
        type=str,
        default="nbertagnolli/counsel-chat",
        help="HuggingFace dataset name for CounselChat"
    )
    
    parser.add_argument(
        "--annomi_csv_path",
        type=str,
        default="./AnnoMI-full.csv",
        help="Path to AnnoMI CSV file"
    )
    
    parser.add_argument(
        "--curriculum_strategy",
        type=str,
        default="therapeutic",
        choices=["therapeutic", "adaptive"],
        help="Curriculum learning strategy"
    )
    
    parser.add_argument(
        "--target_dataset_size",
        type=int,
        default=2000,
        help="Target dataset size per phase"
    )
    
    parser.add_argument(
        "--quality_filter",
        type=str,
        default="high",
        choices=["all", "high", "medium"],
        help="Filter AnnoMI data by MI quality"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced dataset sizes"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without actual training"
    )
    
    parser.add_argument(
        "--run_dpo_after",
        action="store_true",
        help="Run DPO training after curriculum learning"
    )
    
    parser.add_argument(
        "--merge_final_model",
        action="store_true",
        help="Merge final model for deployment"
    )
    
    return parser.parse_args()


def load_datasets(args, logger):
    """Load and prepare datasets for curriculum learning."""
    
    logger.info("=" * 60)
    logger.info("LOADING DATASETS FOR CURRICULUM LEARNING")
    logger.info("=" * 60)
    
    # Load configurations
    config_manager.config_dir = Path(args.config_dir)
    model_config = config_manager.load_config("model_config")
    training_config = config_manager.load_config("training_config")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(
        tokenizer=tokenizer,
        config=training_config,
        logger=logger
    )
    
    # Load CounselChat
    logger.info("Loading CounselChat dataset...")
    counselchat_processor = CounselChatProcessor(
        dataset_name=args.counselchat_dataset,
        logger=logger
    )
    
    raw_counselchat = counselchat_processor.load_raw_data()
    
    if args.debug:
        logger.info("Debug mode: Using reduced CounselChat dataset")
        raw_counselchat = raw_counselchat.select(range(min(200, len(raw_counselchat))))
    
    counselchat_conversations = counselchat_processor.process_conversations(raw_counselchat)
    logger.info(f"Loaded {len(counselchat_conversations)} CounselChat conversations")
    
    # Load AnnoMI
    logger.info("Loading AnnoMI dataset...")
    annomi_path = Path(args.annomi_csv_path)
    
    if not annomi_path.exists():
        logger.error(f"AnnoMI CSV file not found: {annomi_path}")
        return None, None, None
    
    annomi_processor = AnnoMIProcessor(
        csv_path=str(annomi_path),
        logger=logger
    )
    
    raw_annomi = annomi_processor.load_raw_data()
    
    # Apply quality filter
    if args.quality_filter != "all":
        logger.info(f"Applying quality filter: {args.quality_filter}")
        if args.quality_filter == "high":
            raw_annomi = raw_annomi[raw_annomi['mi_quality'] == 'high']
        elif args.quality_filter == "medium":
            raw_annomi = raw_annomi[raw_annomi['mi_quality'].isin(['high', 'medium'])]
    
    if args.debug:
        logger.info("Debug mode: Using reduced AnnoMI dataset")
        raw_annomi = raw_annomi.head(min(100, len(raw_annomi)))
    
    annomi_conversations = annomi_processor.process_conversations(raw_annomi)
    logger.info(f"Loaded {len(annomi_conversations)} AnnoMI conversations")
    
    # Log dataset characteristics
    logger.info("\nDataset Analysis:")
    logger.info(f"  CounselChat: {len(counselchat_conversations)} single-turn conversations")
    logger.info(f"  AnnoMI: {len(annomi_conversations)} multi-turn conversations")
    
    if annomi_conversations:
        avg_turns = sum(len(conv.turns) for conv in annomi_conversations) / len(annomi_conversations)
        logger.info(f"  Average AnnoMI conversation length: {avg_turns:.1f} turns")
    
    ratio = len(counselchat_conversations) / len(annomi_conversations) if annomi_conversations else 0
    logger.info(f"  Volume ratio (CC:AM): {ratio:.1f}:1")
    
    if ratio > 10:
        logger.info("  ✓ Large volume imbalance detected - curriculum learning recommended")
    
    return counselchat_conversations, annomi_conversations, dataset_manager


def main():
    """Main curriculum learning function."""
    args = parse_arguments()
    
    # Setup logging
    logger = Logger("CurriculumTraining").get_logger()
    logger.info("=" * 100)
    logger.info("GEMMA 3N CURRICULUM LEARNING TRAINING")
    logger.info("=" * 100)
    
    start_time = time.time()
    
    # Set random seeds
    set_random_seeds(args.seed)
    logger.info(f"Set random seed to: {args.seed}")
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Log configuration
    logger.info(f"Curriculum strategy: {args.curriculum_strategy}")
    logger.info(f"Target dataset size per phase: {args.target_dataset_size}")
    logger.info(f"Quality filter: {args.quality_filter}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Dry run: {args.dry_run}")
    
    try:
        # Load datasets
        counselchat_conversations, annomi_conversations, dataset_manager = load_datasets(args, logger)
        
        if not counselchat_conversations or not annomi_conversations:
            logger.error("Failed to load required datasets")
            return 1
        
        # Load configurations
        config_manager.config_dir = Path(args.config_dir)
        model_config = config_manager.load_config("model_config")
        training_config = config_manager.load_config("training_config")
        lora_config = config_manager.load_config("lora_config")
        
        # Create output directory
        output_dir = create_output_directory(args.output_dir, args.experiment_name)
        logger.info(f"Output directory: {output_dir}")
        
        if args.dry_run:
            logger.info("Dry run mode - skipping actual training")
            logger.info("Curriculum phases would be:")
            
            # Show what phases would be created
            from curriculum_learning import create_therapeutic_curriculum
            curriculum_manager = create_therapeutic_curriculum(
                counselchat_conversations=counselchat_conversations,
                annomi_conversations=annomi_conversations,
                dataset_manager=dataset_manager,
                strategy_type=args.curriculum_strategy,
                logger=logger
            )
            
            phases = curriculum_manager.strategy.get_phases()
            for phase in phases:
                logger.info(f"  Phase {phase.phase_id}: {phase.name} - "
                           f"CC:{phase.counselchat_ratio:.1%}, AM:{phase.annomi_ratio:.1%}, "
                           f"LR:{phase.learning_rate:.0e}, Epochs:{phase.epochs}")
            
            return 0
        
        # Initialize curriculum training pipeline
        logger.info("Initializing curriculum training pipeline...")
        pipeline = CurriculumTrainingPipeline(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            logger=logger
        )
        
        # Setup curriculum learning
        pipeline.setup_curriculum_learning(
            counselchat_conversations=counselchat_conversations,
            annomi_conversations=annomi_conversations,
            dataset_manager=dataset_manager,
            strategy_type=args.curriculum_strategy
        )
        
        # Run curriculum training
        logger.info("Starting curriculum learning training...")
        curriculum_results = pipeline.run_curriculum_training(
            output_dir=str(output_dir),
            target_dataset_size=args.target_dataset_size
        )
        
        # Run DPO if requested
        if args.run_dpo_after and curriculum_results['final_model_path']:
            logger.info("Running DPO training after curriculum learning...")
            
            # Create combined dataset for DPO
            all_conversations = counselchat_conversations + annomi_conversations
            dpo_dict = {'combined': all_conversations}
            dpo_datasets = dataset_manager.create_training_datasets(dpo_dict, stage="dpo")
            
            # Run DPO
            dpo_result = pipeline.run_dpo(
                train_dataset=dpo_datasets['train'],
                eval_dataset=dpo_datasets['validation'],
                sft_model_path=curriculum_results['final_model_path'],
                output_dir=str(output_dir / "dpo_final")
            )
            
            curriculum_results['dpo_result'] = dpo_result
            curriculum_results['final_model_path'] = dpo_result['output_dir']
        
        # Merge final model if requested
        if args.merge_final_model and curriculum_results['final_model_path']:
            logger.info("Merging final model for deployment...")
            
            merge_output_dir = output_dir / "final_merged_model"
            
            # Import and run merge functionality
            from scripts.merge_adapters import main as merge_main
            import sys
            
            original_argv = sys.argv
            sys.argv = [
                'merge_adapters.py',
                '--adapter_path', curriculum_results['final_model_path'],
                '--output_dir', str(merge_output_dir),
                '--test_generation'
            ]
            
            try:
                merge_result = merge_main()
                if merge_result == 0:
                    curriculum_results['merged_model_path'] = str(merge_output_dir)
                    logger.info(f"Model merged successfully: {merge_output_dir}")
            finally:
                sys.argv = original_argv
        
        # Log final results
        total_time = time.time() - start_time
        
        logger.info("=" * 100)
        logger.info("CURRICULUM LEARNING COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)
        logger.info(f"Total training time: {format_time(total_time)}")
        logger.info(f"Phases completed: {curriculum_results['total_phases_completed']}")
        logger.info(f"Final model path: {curriculum_results['final_model_path']}")
        
        if 'merged_model_path' in curriculum_results:
            logger.info(f"Merged model path: {curriculum_results['merged_model_path']}")
        
        logger.info(f"Results saved to: {output_dir}")
        
        # Log phase summary
        logger.info("\nPhase Summary:")
        for phase_id, phase_result in curriculum_results['phase_results'].items():
            logger.info(f"  Phase {phase_id}: Loss {phase_result['training_loss']:.4f}, "
                       f"Time {format_time(phase_result['training_time'])}")
        
        logger.info("=" * 100)
        
        return 0
        
    except Exception as e:
        logger.error(f"Curriculum training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 