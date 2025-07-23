#!/usr/bin/env python
"""
Gemma-3-4B Supervised Fine-Tuning with Unsloth QLoRA

Phase-wise usage examples
-------------------------
# Phase 1 (General & Emotion)
python run_sft.py \
  --model_name google/gemma-3-4b-it \
  --data_path ./gemma_finetune_datasets/phase1 \
  --output_dir ./checkpoints/phase1

# Phase 2 (Therapeutic) – resume
python run_sft.py \
  --model_name google/gemma-3-4b-it \
  --data_path ./gemma_finetune_datasets/phase2 \
  --resume_from ./checkpoints/phase1 \
  --output_dir ./checkpoints/phase2
"""
from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse
import os
import random
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, set_seed
from peft import TaskType
from trl import SFTTrainer



# ────────────────────────── Helpers ──────────────────────────
def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser("Gemma-3-4B QLoRA SFT")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it")
    parser.add_argument("--data_path", required=True,
                        help="HF Dataset-dict folder (train / validation).")
    parser.add_argument("--output_dir", required=True,
                        help="Output checkpoint directory.")
    parser.add_argument("--resume_from", default=None,
                        help="Resume checkpoint path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--report_to", default="none",
                        choices=["none", "wandb", "tensorboard"])
    return parser.parse_args()


# ────────────────────────── Main ──────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    print("Running with dtype:", dtype)

    # 1. Load Tokenizer and then Gemma 3 4B in 4-bit
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_len,
        dtype=dtype,
        load_in_4bit=True,
    )

    # 2. Attach LoRA adapters
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
    )
    model.print_trainable_parameters()

    # 3. Load dataset
    train_ds = load_from_disk(args.data_path)["train"]

    # 4. TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        report_to=args.report_to,
        seed=args.seed,
    )

    # 5. Trainer with native Unsloth packing
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=targs,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        packing=True, # Use Unsloth's efficient packing
    )

    # 6. Train
    print("Starting SFT…")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 7. Save LoRA adapters & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished. Adapters saved to:", args.output_dir)


if __name__ == "__main__":
    main()