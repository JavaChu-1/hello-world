#!/usr/bin/env python3
"""
Qwen3 7B Fine-tuning Script with LoRA support.
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    use_flash_attention: bool = field(default=True)
    model_max_length: int = field(default=2048)


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to training data (JSONL)"})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data"})
    preprocessing_num_workers: int = field(default=4)


@dataclass
class LoRAArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = field(default="none")
    load_in_4bit: bool = field(default=True)
    load_in_8bit: bool = field(default=False)
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_use_double_quant: bool = field(default=True)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_conversation(messages: List[Dict]) -> str:
    """
    Format conversation messages for Qwen3.

    Qwen uses the ChatML format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """
    formatted = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    return formatted


def preprocess_data(examples, tokenizer, max_length=2048):
    """
    Preprocess data for training.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for messages in examples["messages"]:
        # Format conversation
        formatted_text = format_conversation(messages)

        # Find the position where assistant's response starts
        assistant_start = formatted_text.rfind("<|im_start|>assistant\n")

        if assistant_start == -1:
            continue

        # Tokenize
        full_tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # Tokenize only the prompt part (before assistant's response)
        prompt_text = formatted_text[:assistant_start]
        prompt_tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        # Create labels: -100 for prompt tokens, actual token ids for completion
        labels = [-100] * len(prompt_tokenized["input_ids"]) + \
                 input_ids[len(prompt_tokenized["input_ids"]):]

        # Ensure all sequences have the same length as input_ids
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))
        labels = labels[:len(input_ids)]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up model arguments
    model_args = ModelArguments(
        model_name_or_path=config.get("model_name_or_path"),
        use_flash_attention=config.get("use_flash_attention", True),
        model_max_length=config.get("model_max_length", 2048),
    )

    # Set up data arguments
    data_args = DataArguments(
        data_path=config.get("data_path"),
        val_data_path=config.get("val_data_path"),
        preprocessing_num_workers=config.get("preprocessing_num_workers", 4),
    )

    # Set up LoRA arguments
    lora_args = LoRAArguments(
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        lora_target_modules=config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_bias=config.get("lora_bias", "none"),
        load_in_4bit=config.get("load_in_4bit", True),
        load_in_8bit=config.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=config.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./outputs"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        optim=config.get("optim", "paged_adamw_32bit"),
        logging_dir=config.get("logging_dir", "./outputs/logs"),
        report_to=config.get("report_to", "tensorboard"),
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        local_rank=args.local_rank,
    )

    # Load tokenizer
    print(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=model_args.model_max_length,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up quantization config
    compute_dtype = getattr(torch, lora_args.bnb_4bit_compute_dtype)
    bnb_config = None

    if lora_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=lora_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=lora_args.bnb_4bit_use_double_quant,
        )
    elif lora_args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto" if lora_args.use_lora else None,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "eager",
    )

    # Apply LoRA
    if lora_args.use_lora:
        print("Preparing model for LoRA training")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load datasets
    print(f"Loading training data from {data_args.data_path}")
    train_dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    eval_dataset = None
    if data_args.val_data_path:
        print(f"Loading validation data from {data_args.val_data_path}")
        eval_dataset = load_dataset("json", data_files=data_args.val_data_path, split="train")

    # Preprocess datasets
    print("Preprocessing training data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer, model_args.model_max_length),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
    )

    if eval_dataset:
        print("Preprocessing validation data...")
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_data(x, tokenizer, model_args.model_max_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
        )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()
