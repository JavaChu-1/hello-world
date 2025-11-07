#!/usr/bin/env python3
"""
Data preparation script for Qwen3 fine-tuning.
Converts various data formats to the required JSONL format.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd


def create_conversation_format(instruction: str, input_text: str, output: str) -> Dict:
    """
    Create conversation format for Qwen models.

    Format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def convert_alpaca_format(input_file: str, output_file: str):
    """Convert Alpaca-style JSON to Qwen conversation format."""
    print(f"Converting {input_file} to {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_data = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        conv = create_conversation_format(instruction, input_text, output)
        converted_data.append(conv)

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted_data)} samples to {output_file}")


def convert_csv_format(input_file: str, output_file: str,
                       instruction_col: str = 'instruction',
                       input_col: str = 'input',
                       output_col: str = 'output'):
    """Convert CSV to Qwen conversation format."""
    print(f"Converting {input_file} to {output_file}")

    df = pd.read_csv(input_file)

    converted_data = []
    for _, row in df.iterrows():
        instruction = row.get(instruction_col, '')
        input_text = row.get(input_col, '')
        output = row.get(output_col, '')

        conv = create_conversation_format(instruction, input_text, output)
        converted_data.append(conv)

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted_data)} samples to {output_file}")


def create_sample_data(output_file: str, num_samples: int = 100):
    """Create sample training data for testing."""
    print(f"Creating {num_samples} sample data points...")

    sample_instructions = [
        ("What is machine learning?",
         "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
        ("Explain the difference between supervised and unsupervised learning.",
         "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."),
        ("What is a neural network?",
         "A neural network is a computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information."),
        ("Describe what deep learning is.",
         "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data."),
        ("What is natural language processing?",
         "Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language."),
    ]

    samples = []
    for i in range(num_samples):
        instruction, output = sample_instructions[i % len(sample_instructions)]

        # Add variation
        if i % 3 == 0:
            instruction = f"请回答：{instruction}"
        elif i % 3 == 1:
            instruction = f"Question: {instruction}"

        conv = create_conversation_format(instruction, "", output)
        samples.append(conv)

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {len(samples)} samples in {output_file}")


def split_train_val(input_file: str, train_file: str, val_file: str,
                    val_ratio: float = 0.1, seed: int = 42):
    """Split data into train and validation sets."""
    print(f"Splitting {input_file} into train and validation sets...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Shuffle
    import random
    random.seed(seed)
    random.shuffle(data)

    # Split
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]

    # Write train
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Write validation
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Qwen3 fine-tuning")
    parser.add_argument('--mode', type=str, required=True,
                       choices=['alpaca', 'csv', 'sample', 'split'],
                       help='Data preparation mode')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--train_output', type=str, help='Train output file path')
    parser.add_argument('--val_output', type=str, help='Validation output file path')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of sample data points to create')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--instruction_col', type=str, default='instruction',
                       help='Column name for instruction (CSV mode)')
    parser.add_argument('--input_col', type=str, default='input',
                       help='Column name for input (CSV mode)')
    parser.add_argument('--output_col', type=str, default='output',
                       help='Column name for output (CSV mode)')

    args = parser.parse_args()

    if args.mode == 'alpaca':
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for alpaca mode")
        convert_alpaca_format(args.input, args.output)

    elif args.mode == 'csv':
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for csv mode")
        convert_csv_format(args.input, args.output,
                          args.instruction_col, args.input_col, args.output_col)

    elif args.mode == 'sample':
        if not args.output:
            raise ValueError("--output is required for sample mode")
        create_sample_data(args.output, args.num_samples)

    elif args.mode == 'split':
        if not args.input or not args.train_output or not args.val_output:
            raise ValueError("--input, --train_output, and --val_output are required for split mode")
        split_train_val(args.input, args.train_output, args.val_output, args.val_ratio)

    print("Data preparation completed!")


if __name__ == "__main__":
    main()
