#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi4-mini (Phi4Flash) Demo Inference Script

This script demonstrates how to load and run inference with the Phi4-mini model
using PaddlePaddle and PaddleFormers.

Usage:
    python demo_phi4mini_inference.py --model_path ./phi4mini
    python demo_phi4mini_inference.py --model_path ./phi4mini --prompt "Hello, how are you?"
"""

import argparse
import copy
import json
import os
import sys

import paddle

# Add PaddleFormers to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paddleformers.transformers.auto.modeling import AutoModelForCausalLM
from paddleformers.transformers.auto.tokenizer import AutoTokenizer


def patch_config_json(model_path: str) -> None:
    """
    Patch the config.json in place to change model_type from 'phi4flash' to 'phi4'.
    This allows PaddleFormers to recognize and load the model correctly.

    Args:
        model_path: Path to the model directory
    """
    config_path = os.path.join(model_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Only patch if model_type is phi4flash and hasn't been patched yet
    if config.get("model_type") == "phi4flash":
        print(f"Detected phi4flash model, patching config.json for phi4 compatibility...")

        # Create a backup
        backup_path = config_path + ".backup"
        with open(backup_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Backed up original config to: {backup_path}")

        # Patch the model_type
        config["model_type"] = "phi4"

        # Write the patched config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Patched config.json successfully")


def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16"):
    """
    Load the Phi4-mini model and tokenizer from the specified path.

    Args:
        model_path: Path to the model directory containing config.json, tokenizer files, and safetensors weights
        dtype: Data type for model weights ('float16', 'bfloat16', or 'float32')

    Returns:
        model: Loaded Phi4ForCausalLM model
        tokenizer: Loaded tokenizer
    """
    print(f"Loading model from: {model_path}")

    # Patch config if needed (phi4flash -> phi4)
    patch_config_json(model_path)

    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Tokenizer loaded successfully")
        print(f"  - vocab_size: {len(tokenizer)}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}: {e}")
        print("Trying to load with fallback...")
        try:
            # Try using transformers directly as fallback
            from transformers import AutoTokenizer as HFTokenizer
            tokenizer = HFTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"Tokenizer loaded successfully via transformers")
        except Exception as e2:
            raise RuntimeError(f"Failed to load tokenizer: {e2}")

    # Load model using from_pretrained (handles safetensors automatically)
    print("\nLoading model...")
    print(f"  dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
    )

    print("Model loaded successfully!")
    print(f"  - model_type: {model.config.model_type}")
    print(f"  - vocab_size: {model.config.vocab_size}")
    print(f"  - hidden_size: {model.config.hidden_size}")
    print(f"  - num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  - num_attention_heads: {model.config.num_attention_heads}")
    print(f"  - num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"  - max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"  - mb_per_layer: {model.config.mb_per_layer}")
    print(f"  - sliding_window: {model.config.sliding_window[:5]}... (first 5 layers)")

    # Set model to eval mode
    model.eval()

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """
    Generate text using the Phi4-mini model.

    Args:
        model: The Phi4ForCausalLM model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        Generated text string
    """
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Set padding_side to left for Flash Attention compatibility
    if hasattr(tokenizer, 'padding_side') and tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pd")
    input_ids = inputs["input_ids"]

    print(f"Input token ids shape: {input_ids.shape}")

    # Generate
    print(f"\nGenerating {max_new_tokens} tokens...")
    print(f"Parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print("-" * 60)

    with paddle.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            decode_strategy="sampling",
            use_cache=False,  # Disable cache to avoid dtype mismatch issues with Mamba layers
        )

    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Phi4-mini Demo Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./phi4mini",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you today?",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Phi4-mini Demo Inference")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype)

    # Generate text
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
    )

    print("\n" + "=" * 60)
    print("Generated Text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
