#!/usr/bin/env python3
"""
Adapter Merging Script
Merge LoRA adapters with base model for deployment and create final merged model.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import Logger, ModelUtils, get_device_info
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained adapter model (DPO output)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./final_model",
        help="Output directory for merged model"
    )
    
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="google/gemma-3n-E2B-it",
        help="Base model name or path"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--hub_model_name",
        type=str,
        help="Model name for Hugging Face Hub (required if push_to_hub)"
    )
    
    parser.add_argument(
        "--hub_token",
        type=str,
        help="Hugging Face Hub token (or set HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--test_generation",
        action="store_true",
        help="Test generation with merged model"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading"
    )
    
    return parser.parse_args()


def test_model_generation(model, tokenizer, logger):
    """Test the merged model with sample therapeutic prompts."""
    
    test_prompts = [
        "I've been feeling really anxious lately and I don't know what to do about it.",
        "I'm struggling with depression and feel like nothing I do matters.",
        "I want to quit smoking but I keep relapsing. What should I do?",
        "My relationship is falling apart and I don't know how to fix it."
    ]
    
    logger.info("Testing model generation with sample prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n--- Test {i} ---")
        logger.info(f"Prompt: {prompt}")
        
        # Format as conversation
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Response: {generated_text.strip()}")
            
        except Exception as e:
            logger.error(f"Generation failed for prompt {i}: {e}")


def main():
    """Main merging function."""
    args = parse_arguments()
    
    # Setup logging
    logger = Logger("AdapterMerging").get_logger()
    logger.info("=" * 80)
    logger.info("GEMMA 3N ADAPTER MERGING")
    logger.info("=" * 80)
    
    # Log device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Validate adapter path
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {adapter_path}")
        return 1
    
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Base model: {args.base_model_name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load PEFT config
        logger.info("Loading PEFT configuration...")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        logger.info(f"PEFT config: {peft_config}")
        
        # Load base model
        logger.info(f"Loading base model: {args.base_model_name}")
        torch_dtype = getattr(torch, args.torch_dtype)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Base model loaded successfully")
        ModelUtils.print_model_info(base_model, logger)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load PEFT model
        logger.info("Loading PEFT model...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("PEFT model loaded successfully")
        
        # Merge adapters
        logger.info("Merging adapters with base model...")
        merged_model = peft_model.merge_and_unload()
        logger.info("Adapters merged successfully")
        
        # Print merged model info
        ModelUtils.print_model_info(merged_model, logger)
        
        # Test generation if requested
        if args.test_generation:
            test_model_generation(merged_model, tokenizer, logger)
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_dir}")
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        tokenizer.save_pretrained(output_dir)
        
        # Save model info
        model_info = {
            "base_model": args.base_model_name,
            "adapter_path": str(adapter_path),
            "merge_date": str(torch.utils.data.get_worker_info()),
            "torch_dtype": args.torch_dtype,
            "device_info": device_info
        }
        
        import json
        with open(output_dir / "merge_info.json", "w") as f:
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info("Merged model saved successfully")
        
        # Push to Hub if requested
        if args.push_to_hub:
            if not args.hub_model_name:
                logger.error("hub_model_name is required when push_to_hub is True")
                return 1
            
            logger.info(f"Pushing model to Hugging Face Hub: {args.hub_model_name}")
            
            # Set up authentication
            hub_token = args.hub_token or os.getenv("HF_TOKEN")
            if hub_token:
                from huggingface_hub import login
                login(token=hub_token)
            
            try:
                # Push model
                merged_model.push_to_hub(
                    args.hub_model_name,
                    private=False,
                    commit_message="Merged Gemma 3n therapeutic conversation model"
                )
                
                # Push tokenizer
                tokenizer.push_to_hub(
                    args.hub_model_name,
                    commit_message="Tokenizer for merged Gemma 3n therapeutic model"
                )
                
                logger.info(f"Model successfully pushed to: https://huggingface.co/{args.hub_model_name}")
                
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")
                return 1
        
        # Final success message
        logger.info("=" * 80)
        logger.info("ADAPTER MERGING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Merged model saved to: {output_dir}")
        if args.push_to_hub:
            logger.info(f"Model available on Hub: https://huggingface.co/{args.hub_model_name}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Adapter merging failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
