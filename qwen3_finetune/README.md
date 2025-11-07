# Qwen3 7B Fine-tuning Project

This project provides a complete setup for fine-tuning the Qwen3 7B model using LoRA (Low-Rank Adaptation) or full fine-tuning approaches.

## Features

- LoRA fine-tuning with 4-bit/8-bit quantization support
- Full parameter fine-tuning with DeepSpeed ZeRO-3
- Flexible data processing pipeline
- Interactive inference script
- Easy-to-use configuration system
- Support for distributed training

## Project Structure

```
qwen3_finetune/
├── configs/
│   ├── lora_config.yaml              # LoRA training configuration
│   ├── full_finetune_config.yaml    # Full fine-tuning configuration
│   └── ds_config_zero3.json         # DeepSpeed ZeRO-3 configuration
├── data/
│   └── example.jsonl                # Example training data
├── scripts/
│   ├── prepare_data.py              # Data preparation script
│   ├── train.py                     # Main training script
│   ├── inference.py                 # Inference script
│   └── run_training.sh              # Quick start training script
├── models/                          # Downloaded models (optional)
├── outputs/                         # Training outputs
└── requirements.txt                 # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
cd qwen3_finetune
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Flash Attention 2 (Optional but recommended)

```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

### Data Format

Training data should be in JSONL format with conversation messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Create Sample Data

```bash
python scripts/prepare_data.py \
    --mode sample \
    --output data/sample.jsonl \
    --num_samples 100
```

### Convert from Alpaca Format

```bash
python scripts/prepare_data.py \
    --mode alpaca \
    --input your_alpaca_data.json \
    --output data/converted.jsonl
```

### Convert from CSV

```bash
python scripts/prepare_data.py \
    --mode csv \
    --input your_data.csv \
    --output data/converted.jsonl \
    --instruction_col "instruction" \
    --input_col "input" \
    --output_col "output"
```

### Split Train/Validation

```bash
python scripts/prepare_data.py \
    --mode split \
    --input data/all_data.jsonl \
    --train_output data/train.jsonl \
    --val_output data/val.jsonl \
    --val_ratio 0.1
```

## Training

### Quick Start with LoRA

1. Prepare your training data and place it in `data/train.jsonl`

2. Edit `configs/lora_config.yaml` if needed:
   - Update `model_name_or_path` (default: "Qwen/Qwen2.5-7B-Instruct")
   - Update `data_path` and `val_data_path`
   - Adjust training hyperparameters

3. Run training:

```bash
bash scripts/run_training.sh
```

Or directly:

```bash
python scripts/train.py --config configs/lora_config.yaml
```

### Full Fine-tuning with DeepSpeed

For full parameter fine-tuning (requires more GPU memory):

```bash
deepspeed --num_gpus=4 scripts/train.py \
    --config configs/full_finetune_config.yaml
```

### Configuration Parameters

Key parameters in `configs/lora_config.yaml`:

- **Model Settings:**
  - `model_name_or_path`: Model to fine-tune
  - `model_max_length`: Maximum sequence length (default: 2048)
  - `use_flash_attention`: Enable Flash Attention 2

- **LoRA Settings:**
  - `lora_r`: LoRA rank (default: 64)
  - `lora_alpha`: LoRA alpha (default: 16)
  - `lora_dropout`: Dropout rate (default: 0.05)
  - `load_in_4bit`: Use 4-bit quantization

- **Training Settings:**
  - `num_train_epochs`: Number of training epochs
  - `per_device_train_batch_size`: Batch size per device
  - `gradient_accumulation_steps`: Gradient accumulation steps
  - `learning_rate`: Learning rate (default: 2e-4)

## Inference

### Interactive Chat Mode

```bash
python scripts/inference.py \
    --model_path outputs \
    --lora_path outputs \
    --load_in_4bit \
    --interactive
```

### Single Query

```bash
python scripts/inference.py \
    --model_path outputs \
    --lora_path outputs \
    --load_in_4bit \
    --prompt "What is machine learning?"
```

### Advanced Inference Options

```bash
python scripts/inference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --lora_path outputs/checkpoint-1000 \
    --load_in_4bit \
    --interactive \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --max_new_tokens 512
```

## Hardware Requirements

### LoRA Fine-tuning (Recommended)

- **GPU Memory:** 16GB+ (single GPU)
- **Recommended:** 24GB+ for larger batch sizes
- **Example:** RTX 3090, RTX 4090, A10, A100 (40GB)

With 4-bit quantization:
- Can run on 12GB GPUs (e.g., RTX 3060)

### Full Fine-tuning

- **GPU Memory:** 80GB+ (multiple GPUs)
- **Recommended:** 4x A100 80GB
- Requires DeepSpeed ZeRO-3 optimization

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

### Weights & Biases

To use W&B, update `configs/lora_config.yaml`:

```yaml
report_to: "wandb"
```

And set your API key:

```bash
export WANDB_API_KEY=your_api_key
```

## Tips for Better Results

1. **Data Quality:** Use high-quality, diverse training data
2. **Data Size:** Aim for at least 1,000-10,000 examples
3. **Learning Rate:** Start with 2e-4 for LoRA, 5e-6 for full fine-tuning
4. **LoRA Rank:** Higher rank (64-128) for complex tasks, lower (8-16) for simple tasks
5. **Evaluation:** Always use a validation set to monitor overfitting
6. **Gradient Accumulation:** Increase if you run out of memory
7. **Mixed Precision:** Use bf16 on Ampere+ GPUs for stability

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use 4-bit quantization
- Reduce `model_max_length`

### Slow Training

- Enable Flash Attention 2
- Increase `per_device_train_batch_size`
- Reduce `dataloader_num_workers` if CPU-bound
- Use multiple GPUs with `torchrun` or `deepspeed`

### Model Not Learning

- Check data format and preprocessing
- Increase learning rate
- Train for more epochs
- Increase LoRA rank
- Check if labels are correctly set

## Common Commands

### Distributed Training (Multiple GPUs)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/lora_config.yaml
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    --config configs/lora_config.yaml \
    --resume_from_checkpoint outputs/checkpoint-500
```

### Merge LoRA Weights

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "outputs")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
```

## Resources

- [Qwen2.5 Documentation](https://github.com/QwenLM/Qwen2.5)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use Qwen models, please cite:

```bibtex
@article{qwen2.5,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  year={2024}
}
```
