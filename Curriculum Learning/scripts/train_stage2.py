#!/usr/bin/env python3
"""
Stage 2 Training Script: AnnoMI SFT
Fine-tune Stage 1 model on AnnoMI dataset for specialized motivational interviewing.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    config_manager, Logger, set_random_seeds, 
    create_output_directory, get_device_info
)
from data_preprocessing import (
    AnnoMIProcessor, DatasetManager
)
from training import SFTTrainerWrapper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 2 Training: AnnoMI SFT")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints/stage2",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--stage1_model_path",
        type=str,
        required=True,
        help="Path to Stage 1 trained model"
    )
    
    parser.add_argument(
        "--annomi_csv_path",
        type=str,
        default="./AnnoMI-full.csv",
        help="Path to AnnoMI CSV file"
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
        default="gemma3n_stage2_annomi",
        help="Name for the experiment"
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
        help="Enable debug mode with reduced dataset size"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without actual training"
    )
    
    parser.add_argument(
        "--quality_filter",
        type=str,
        default="high",
        choices=["all", "high", "medium"],
        help="Filter AnnoMI data by MI quality"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    logger = Logger("Stage2Training").get_logger()
    logger.info("=" * 80)
    logger.info("GEMMA 3N STAGE 2 TRAINING: ANNOMI SFT")
    logger.info("=" * 80)
    
    # Set random seeds
    set_random_seeds(args.seed)
    logger.info(f"Set random seed to: {args.seed}")
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Validate Stage 1 model path
    stage1_path = Path(args.stage1_model_path)
    if not stage1_path.exists():
        logger.error(f"Stage 1 model path does not exist: {stage1_path}")
        return 1
    
    logger.info(f"Stage 1 model path: {stage1_path}")
    
    # Validate AnnoMI CSV path
    annomi_path = Path(args.annomi_csv_path)
    if not annomi_path.exists():
        logger.error(f"AnnoMI CSV file does not exist: {annomi_path}")
        return 1
    
    logger.info(f"AnnoMI CSV path: {annomi_path}")
    
    # Load configurations
    try:
        config_manager.config_dir = Path(args.config_dir)
        model_config = config_manager.load_config("model_config")
        training_config = config_manager.load_config("training_config")
        lora_config = config_manager.load_config("lora_config")
        
        logger.info("Loaded all configuration files")
        
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return 1
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir, args.experiment_name)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize dataset processor
        logger.info("Initializing AnnoMI dataset processor...")
        annomi_processor = AnnoMIProcessor(
            csv_path=str(annomi_path),
            logger=logger
        )
        
        # Load and process dataset
        logger.info("Loading and processing AnnoMI dataset...")
        raw_data = annomi_processor.load_raw_data()
        
        # Apply quality filter
        if args.quality_filter != "all":
            logger.info(f"Applying quality filter: {args.quality_filter}")
            if args.quality_filter == "high":
                raw_data = raw_data[raw_data['mi_quality'] == 'high']
            elif args.quality_filter == "medium":
                raw_data = raw_data[raw_data['mi_quality'].isin(['high', 'medium'])]
            
            logger.info(f"Filtered dataset size: {len(raw_data)} rows")
        
        # Debug mode: use smaller dataset
        if args.debug:
            logger.info("Debug mode: Using reduced dataset size")
            raw_data = raw_data.head(min(500, len(raw_data)))
        
        # Process conversations
        conversations = annomi_processor.process_conversations(raw_data)
        logger.info(f"Processed {len(conversations)} conversations")
        
        if not conversations:
            logger.error("No valid conversations found in dataset")
            return 1
        
        # Initialize dataset manager
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer']['name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset_manager = DatasetManager(
            tokenizer=tokenizer,
            config=training_config,
            logger=logger
        )
        
        # Create training datasets
        logger.info("Creating training datasets...")
        conversations_dict = {"annomi": conversations}
        datasets = dataset_manager.create_training_datasets(conversations_dict, stage="sft")
        
        # Log dataset statistics
        stats = dataset_manager.get_dataset_statistics(conversations_dict)
        logger.info(f"Dataset statistics: {stats}")
        
        logger.info(f"Training dataset size: {len(datasets['train'])}")
        logger.info(f"Validation dataset size: {len(datasets['validation'])}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully. Exiting without training.")
            return 0
        
        # Initialize trainer
        logger.info("Initializing SFT trainer...")
        trainer = SFTTrainerWrapper(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            logger=logger
        )
        
        # Load Stage 1 model
        logger.info(f"Loading Stage 1 model from: {stage1_path}")
        trainer.load_model(str(stage1_path))
        
        # Start training
        logger.info("Starting Stage 2 SFT training...")
        training_result = trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            output_dir=str(output_dir),
            stage="stage2",
            experiment_name=args.experiment_name
        )
        
        # Log final results
        logger.info("=" * 80)
        logger.info("STAGE 2 TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {training_result['training_loss']:.4f}")
        logger.info(f"Training time: {training_result['training_time']:.2f} seconds")
        logger.info(f"Total steps: {training_result['global_step']}")
        logger.info(f"Model saved to: {training_result['output_dir']}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
