#!/usr/bin/env python
# Filename: run_sft.py

import argparse
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_from_disk
from transformers import set_seed
from peft import TaskType
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# Environment-specific configuration for Hugging Face cache
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Kaggle environment detected. Setting HF_DATASETS_CACHE to /kaggle/working/hf_cache")
    os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/hf_cache"

# ────────────────────────── Helpers ──────────────────────────

def parse_args() -> argparse.Namespace:
    """CLI argument parser with stability-focused defaults."""
    parser = argparse.ArgumentParser(description="Gemma-3 4B QLoRA SFT")
    parser.add_argument("--model_name", default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", help="The model name or path to load from.")
    parser.add_argument("--data_path", required=True, help="Path to the preprocessed dataset folder (e.g., /path/to/phase1).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final LoRA adapters and tokenizer.")
    parser.add_argument("--resume_from", default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate. 2e-5 is a stable default for fine-tuning.")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save a checkpoint every N steps.")
    parser.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"], help="Logging and reporting service.")
    return parser.parse_args()

# ────────────────────────── Main Execution ──────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required to run this script.")
    
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    print(f"Running with dtype: {dtype}")

    # Step 1: Load Tokenizer and Add Custom Tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    domain_tags = [f"<domain>{domain}</domain>" for domain in ["general", "chat", "emotion", "therapeutic"]]
    tokenizer.add_special_tokens({"additional_special_tokens": domain_tags})
    print(f"Added {len(domain_tags)} new domain tags to the tokenizer.")

    # Step 2: Load the Model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_len,
        dtype=dtype,
        load_in_4bit=True,
    )

    # Step 3: Synchronize Model and Tokenizer
    # CRITICAL: Resize the model's token embeddings to match the new tokenizer size.
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model token embeddings to {len(tokenizer)}.")

    # Step 4: Apply the Official Gemma-3 Chat Template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    # Step 4: Configure LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    # Step 5: Load the Dataset
    train_ds = load_from_disk(args.data_path)["train"]

    # Step 6: Configure Training Arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        gradient_checkpointing=True,
        report_to=args.report_to,
        seed=args.seed,
        dataset_text_field="text",
        max_grad_norm=0.3,
        max_seq_length=args.max_len,
    )

    # Step 7: Instantiate the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
    )

    # Step 8: Configure Loss Masking
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Step 9: Start the Fine-Tuning Process
    print("Starting Supervised Fine-Tuning...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Step 10: Save the Final LoRA Adapters and Tokenizer
    print("Training finished. Saving adapters and tokenizer to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model adapters and tokenizer saved successfully.")

if __name__ == "__main__":
    main()
