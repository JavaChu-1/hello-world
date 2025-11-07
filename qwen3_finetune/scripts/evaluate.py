#!/usr/bin/env python3
"""
Evaluation script for fine-tuned models.
Evaluates model on test set and calculates metrics.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


def load_model_and_tokenizer(model_path, lora_path=None):
    """Load model and tokenizer."""
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


def format_messages(messages):
    """Format messages in ChatML format."""
    formatted = ""
    for message in messages[:-1]:  # Exclude last assistant message
        role = message["role"]
        content = message["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate response for given messages."""
    formatted_input = format_messages(messages)
    inputs = tokenizer(formatted_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def calculate_perplexity(model, tokenizer, dataset, max_samples=None):
    """Calculate perplexity on dataset."""
    total_loss = 0
    total_tokens = 0
    num_samples = 0

    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    print("Calculating perplexity...")
    for item in tqdm(samples):
        messages = item["messages"]

        # Format full conversation
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Tokenize
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Calculate loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        # Accumulate
        total_loss += loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
        num_samples += 1

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item(), num_samples


def evaluate_generation_quality(model, tokenizer, dataset, output_file=None, max_samples=100):
    """Evaluate generation quality and save results."""
    results = []

    samples = dataset.select(range(min(max_samples, len(dataset))))

    print("Evaluating generation quality...")
    for idx, item in enumerate(tqdm(samples)):
        messages = item["messages"]

        # Get reference response (last assistant message)
        reference = messages[-1]["content"]

        # Generate response
        generated = generate_response(model, tokenizer, messages)

        # Store result
        result = {
            "index": idx,
            "input": messages[-2]["content"],  # Last user message
            "reference": reference,
            "generated": generated,
        }
        results.append(result)

    # Save results if output file is specified
    if output_file:
        print(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA adapter")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data (JSONL)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl",
                       help="Output file for generation results")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--calculate_perplexity", action="store_true",
                       help="Calculate perplexity")
    parser.add_argument("--evaluate_generation", action="store_true",
                       help="Evaluate generation quality")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path)

    # Load test dataset
    print(f"Loading test data from {args.test_data}")
    test_dataset = load_dataset("json", data_files=args.test_data, split="train")

    print(f"Test dataset size: {len(test_dataset)}")

    # Calculate perplexity
    if args.calculate_perplexity:
        perplexity, num_samples = calculate_perplexity(
            model, tokenizer, test_dataset, args.max_samples
        )
        print(f"\nPerplexity: {perplexity:.2f} (evaluated on {num_samples} samples)")

    # Evaluate generation quality
    if args.evaluate_generation:
        results = evaluate_generation_quality(
            model, tokenizer, test_dataset, args.output_file, args.max_samples
        )
        print(f"\nEvaluated {len(results)} samples")
        print(f"Results saved to {args.output_file}")

        # Print a few examples
        print("\nExample outputs:")
        for i, result in enumerate(results[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Input: {result['input'][:100]}...")
            print(f"Generated: {result['generated'][:200]}...")
            print(f"Reference: {result['reference'][:200]}...")

    if not args.calculate_perplexity and not args.evaluate_generation:
        print("Please specify at least one evaluation mode:")
        print("  --calculate_perplexity: Calculate perplexity on test set")
        print("  --evaluate_generation: Evaluate generation quality")


if __name__ == "__main__":
    main()
