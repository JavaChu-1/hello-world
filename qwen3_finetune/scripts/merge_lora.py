#!/usr/bin/env python3
"""
Merge LoRA adapter weights with base model.
This creates a standalone model that doesn't require PEFT.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device_map: str = "auto",
):
    """
    Merge LoRA adapter weights with base model.

    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapter
        output_path: Path to save merged model
        device_map: Device map for model loading
    """

    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print(f"Loading and saving tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)

    print("Merge completed successfully!")
    print(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base model (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter (e.g., ./outputs/checkpoint-1000)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Merge weights
    merge_lora_weights(
        args.base_model_path,
        args.lora_adapter_path,
        args.output_path,
        args.device_map,
    )


if __name__ == "__main__":
    main()
