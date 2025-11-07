# Quick Start Guide

This guide will help you get started with fine-tuning Qwen3 7B in just a few steps.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 50GB+ free disk space

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

## Step 2: Prepare Your Data

### Option A: Use Sample Data (for testing)

```bash
python scripts/prepare_data.py \
    --mode sample \
    --output data/all_data.jsonl \
    --num_samples 200

# Split into train and validation
python scripts/prepare_data.py \
    --mode split \
    --input data/all_data.jsonl \
    --train_output data/train.jsonl \
    --val_output data/val.jsonl \
    --val_ratio 0.1
```

### Option B: Use Your Own Data

Prepare your data in JSONL format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your question here"},
    {"role": "assistant", "content": "The answer here"}
  ]
}
```

Save as `data/train.jsonl` and `data/val.jsonl`.

## Step 3: Configure Training

Edit `configs/lora_config.yaml`:

```yaml
# Update these paths if needed
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
data_path: "./data/train.jsonl"
val_data_path: "./data/val.jsonl"
output_dir: "./outputs"

# Adjust based on your GPU
per_device_train_batch_size: 2  # Reduce to 1 if OOM
gradient_accumulation_steps: 8  # Increase if batch size is reduced
```

## Step 4: Start Training

```bash
# Make the script executable
chmod +x scripts/run_training.sh

# Run training
bash scripts/run_training.sh
```

Or run directly:

```bash
python scripts/train.py --config configs/lora_config.yaml
```

Training will:
1. Download the model (first time only)
2. Load and preprocess your data
3. Start training with progress updates
4. Save checkpoints to `outputs/`

## Step 5: Monitor Training

In a new terminal:

```bash
# Activate the same virtual environment
source venv/bin/activate

# Start TensorBoard
tensorboard --logdir outputs/logs
```

Open http://localhost:6006 in your browser.

## Step 6: Test Your Model

### Interactive Chat

```bash
python scripts/inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --lora_path outputs \
    --load_in_4bit \
    --interactive
```

### Single Query

```bash
python scripts/inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --lora_path outputs \
    --load_in_4bit \
    --prompt "What is machine learning?"
```

## Step 7: (Optional) Merge LoRA Weights

Create a standalone model without requiring PEFT:

```bash
python scripts/merge_lora.py \
    --base_model_path Qwen/Qwen2.5-7B-Instruct \
    --lora_adapter_path outputs \
    --output_path merged_model
```

Then use the merged model:

```bash
python scripts/inference.py \
    --model_path merged_model \
    --interactive
```

## Common Issues

### Out of Memory (OOM)

1. Reduce batch size in `configs/lora_config.yaml`:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   ```

2. Reduce sequence length:
   ```yaml
   model_max_length: 1024
   ```

3. Enable 4-bit quantization (should already be enabled):
   ```yaml
   load_in_4bit: true
   ```

### Model Download Issues

If downloading from HuggingFace is slow:

1. Download manually:
   ```bash
   git lfs install
   git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct
   ```

2. Update config:
   ```yaml
   model_name_or_path: "./models/Qwen2.5-7B-Instruct"
   ```

### Training is Slow

1. Install Flash Attention 2:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. Increase batch size if you have GPU memory:
   ```yaml
   per_device_train_batch_size: 4
   ```

3. Use multiple GPUs:
   ```bash
   torchrun --nproc_per_node=2 scripts/train.py --config configs/lora_config.yaml
   ```

## Next Steps

- Read the full [README.md](README.md) for advanced usage
- Experiment with different hyperparameters
- Try different LoRA configurations
- Add more training data for better results
- Evaluate your model using `scripts/evaluate.py`

## Training Time Estimates

With a single RTX 4090 (24GB):
- 1,000 samples: ~30 minutes
- 10,000 samples: ~5 hours
- 100,000 samples: ~2 days

Times vary based on sequence length and batch size.

## Getting Help

If you encounter issues:
1. Check the [README.md](README.md) troubleshooting section
2. Review configuration files for typos
3. Ensure your data is in the correct format
4. Check GPU memory usage with `nvidia-smi`

Happy fine-tuning!
