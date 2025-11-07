#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen3 models.
Supports both base model and LoRA adapter inference.
"""

import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_model_and_tokenizer(
    model_path: str,
    lora_path: str = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_flash_attention: bool = True,
):
    """Load model and tokenizer."""

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Set up quantization config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
    )

    # Load LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference

    model.eval()
    return model, tokenizer


def format_messages(messages):
    """Format messages in ChatML format for Qwen."""
    formatted = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """Generate response for given messages."""

    # Format input
    formatted_input = format_messages(messages)

    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def interactive_chat(model, tokenizer, args):
    """Interactive chat mode."""
    print("\n" + "="*50)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("="*50 + "\n")

    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        # Add user message
        conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        print("Assistant: ", end="", flush=True)
        response = generate_response(
            model,
            tokenizer,
            conversation_history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(response)
        print()

        # Add assistant message
        conversation_history.append({"role": "assistant", "content": response})


def single_inference(model, tokenizer, args):
    """Single inference mode."""
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt}
    ]

    print(f"User: {args.prompt}\n")
    print("Assistant: ", end="", flush=True)

    response = generate_response(
        model,
        tokenizer,
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    print(response)
    print()


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen3 model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to base model or fine-tuned model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA adapter (optional)")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Use Flash Attention 2")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive chat mode")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Prompt for single inference mode")
    parser.add_argument("--system_prompt", type=str,
                       default="You are a helpful assistant.",
                       help="System prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.lora_path,
        args.load_in_4bit,
        args.load_in_8bit,
        args.use_flash_attention,
    )

    # Run inference
    if args.interactive:
        interactive_chat(model, tokenizer, args)
    else:
        single_inference(model, tokenizer, args)


if __name__ == "__main__":
    main()
