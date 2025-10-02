#!/usr/bin/env python3
"""
Preference Tuning Script: DPO Training
Apply Direct Preference Optimization to the SFT model for improved alignment.
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
    CounselChatProcessor, AnnoMIProcessor, DatasetManager
)
from training import DPOTrainerWrapper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DPO Preference Tuning")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints/dpo",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--sft_model_path",
        type=str,
        required=True,
        help="Path to SFT trained model (Stage 2 output)"
    )
    
    parser.add_argument(
        "--annomi_csv_path",
        type=str,
        default="./AnnoMI-full.csv",
        help="Path to AnnoMI CSV file"
    )
    
    parser.add_argument(
        "--counselchat_dataset",
        type=str,
        default="nbertagnolli/counsel-chat",
        help="HuggingFace dataset name for CounselChat"
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
        default="gemma3n_dpo",
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
        "--use_both_datasets",
        action="store_true",
        help="Use both CounselChat and AnnoMI for preference pairs"
    )
    
    parser.add_argument(
        "--preference_ratio",
        type=float,
        default=0.7,
        help="Ratio of AnnoMI to CounselChat data for preference pairs"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    logger = Logger("DPOTraining").get_logger()
    logger.info("=" * 80)
    logger.info("GEMMA 3N DPO PREFERENCE TUNING")
    logger.info("=" * 80)
    
    # Set random seeds
    set_random_seeds(args.seed)
    logger.info(f"Set random seed to: {args.seed}")
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Validate SFT model path
    sft_path = Path(args.sft_model_path)
    if not sft_path.exists():
        logger.error(f"SFT model path does not exist: {sft_path}")
        return 1
    
    logger.info(f"SFT model path: {sft_path}")
    
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
        # Initialize tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer']['name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize dataset manager
        dataset_manager = DatasetManager(
            tokenizer=tokenizer,
            config=training_config,
            logger=logger
        )
        
        # Load datasets for preference pairs
        all_conversations = []
        
        # Load AnnoMI dataset
        logger.info("Loading AnnoMI dataset for preference pairs...")
        annomi_path = Path(args.annomi_csv_path)
        if annomi_path.exists():
            annomi_processor = AnnoMIProcessor(
                csv_path=str(annomi_path),
                logger=logger
            )
            
            raw_annomi = annomi_processor.load_raw_data()
            
            # Filter for high quality only for DPO
            raw_annomi = raw_annomi[raw_annomi['mi_quality'] == 'high']
            logger.info(f"Using {len(raw_annomi)} high-quality AnnoMI rows")
            
            if args.debug:
                raw_annomi = raw_annomi.head(200)
            
            annomi_conversations = annomi_processor.process_conversations(raw_annomi)
            all_conversations.extend(annomi_conversations)
            logger.info(f"Added {len(annomi_conversations)} AnnoMI conversations")
        
        # Load CounselChat dataset if requested
        if args.use_both_datasets:
            logger.info("Loading CounselChat dataset for preference pairs...")
            counselchat_processor = CounselChatProcessor(
                dataset_name=args.counselchat_dataset,
                logger=logger
            )
            
            raw_counselchat = counselchat_processor.load_raw_data()
            
            if args.debug:
                raw_counselchat = raw_counselchat.select(range(min(100, len(raw_counselchat))))
            
            counselchat_conversations = counselchat_processor.process_conversations(raw_counselchat)
            
            # Balance the datasets according to preference ratio
            n_annomi = len(all_conversations)
            n_counselchat_target = int(n_annomi * (1 - args.preference_ratio) / args.preference_ratio)
            
            if len(counselchat_conversations) > n_counselchat_target:
                import random
                counselchat_conversations = random.sample(counselchat_conversations, n_counselchat_target)
            
            all_conversations.extend(counselchat_conversations)
            logger.info(f"Added {len(counselchat_conversations)} CounselChat conversations")
        
        if not all_conversations:
            logger.error("No conversations loaded for DPO training")
            return 1
        
        logger.info(f"Total conversations for DPO: {len(all_conversations)}")
        
        # Create DPO datasets
        logger.info("Creating DPO preference datasets...")
        conversations_dict = {"combined": all_conversations}
        datasets = dataset_manager.create_training_datasets(conversations_dict, stage="dpo")
        
        # Log dataset statistics
        stats = dataset_manager.get_dataset_statistics(conversations_dict)
        logger.info(f"Dataset statistics: {stats}")
        
        logger.info(f"DPO training dataset size: {len(datasets['train'])}")
        logger.info(f"DPO validation dataset size: {len(datasets['validation'])}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully. Exiting without training.")
            return 0
        
        # Initialize DPO trainer
        logger.info("Initializing DPO trainer...")
        trainer = DPOTrainerWrapper(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            logger=logger
        )
        
        # Load SFT model
        logger.info(f"Loading SFT model from: {sft_path}")
        trainer.load_model(str(sft_path))
        
        # Start DPO training
        logger.info("Starting DPO preference tuning...")
        training_result = trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            output_dir=str(output_dir),
            experiment_name=args.experiment_name
        )
        
        # Log final results
        logger.info("=" * 80)
        logger.info("DPO PREFERENCE TUNING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {training_result['training_loss']:.4f}")
        logger.info(f"Training time: {training_result['training_time']:.2f} seconds")
        logger.info(f"Total steps: {training_result['global_step']}")
        logger.info(f"Model saved to: {training_result['output_dir']}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"DPO training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
