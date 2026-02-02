#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi4-mini Demo Inference Script (No Mamba Layers)

This script demonstrates how to load and run inference with the Phi4-mini model
with Mamba layers disabled (using only Attention layers).

Usage:
    python demo_phi4mini_no_mamba.py --model_path ./phi4mini
    python demo_phi4mini_no_mamba.py --model_path ./phi4mini --prompt "Hello, how are you?"
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


def patch_config_json_no_mamba(model_path: str) -> None:
    """
    Patch the config.json to disable Mamba layers.
    """
    config_path = os.path.join(model_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Only patch if mb_per_layer > 0
    if config.get("mb_per_layer", 0) > 0:
        print(f"Disabling Mamba layers (setting mb_per_layer from {config.get('mb_per_layer')} to 0)...")

        # Create a backup
        backup_path = config_path + ".backup"
        if not os.path.exists(backup_path):
            with open(backup_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Backed up original config to: {backup_path}")

        # Patch mb_per_layer
        config["mb_per_layer"] = 0

        # Write the patched config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Patched config.json successfully")


def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16"):
    """
    Load the Phi4-mini model and tokenizer from the specified path.
    """
    print(f"Loading model from: {model_path}")

    # Patch config to disable Mamba layers
    patch_config_json_no_mamba(model_path)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer loaded successfully")
    print(f"  - vocab_size: {len(tokenizer)}")

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
    print(f"  - mb_per_layer: {model.config.mb_per_layer} (Mamba layers disabled)")

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
    """
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Set padding_side to left for compatibility
    if hasattr(tokenizer, 'padding_side'):
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
        )

    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Phi4-mini Demo Inference (No Mamba)")
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
    print("Phi4-mini Demo Inference (No Mamba Layers)")
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
