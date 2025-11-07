#!/bin/bash
# Quick training script for Qwen3 7B with LoRA

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment if needed
# source venv/bin/activate

# Run training
python scripts/train.py \
    --config configs/lora_config.yaml

echo "Training completed!"
