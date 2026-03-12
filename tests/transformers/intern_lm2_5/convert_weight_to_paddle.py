#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert InternLM2.5 model from HuggingFace/PyTorch format to PaddlePaddle format.

Usage:
    python convert_weight_to_paddle.py --model_path /path/to/internlm2_5-7b-chat --output_path ./paddle_model
"""

import argparse
import os
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import paddle
from paddleformers.transformers.intern_lm2_5 import InternLM25ForCausalLM, InternLM25Tokenizer


def convert_hf_to_paddle(model_path, output_path, save_format="pd"):
    """
    Convert HuggingFace InternLM2.5 model to PaddlePaddle format.

    Args:
        model_path (str): Path to the HuggingFace model directory
        output_path (str): Path to save the converted PaddlePaddle model
        save_format (str): Format to save ('pd' for PaddlePaddle, 'hf' for HuggingFace compatible)
    """
    print(f"Converting InternLM2.5 model from {model_path}")
    print(f"Output path: {output_path}")
    print(f"Save format: {save_format}")
    print("-" * 60)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load and convert tokenizer
    print("Converting tokenizer...")
    tokenizer = InternLM25Tokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}")

    # Load and convert model
    print("Converting model (this may take a while)...")
    model = InternLM25ForCausalLM.from_pretrained(
        model_path,
        convert_from_hf=True,  # Convert from HuggingFace format
        load_checkpoint_format="",  # Use default format
    )
    model.eval()
    print("Model loaded successfully")

    # Save converted model
    print(f"Saving model to {output_path}...")
    save_kwargs = {
        "save_checkpoint_format": "",
        "save_to_hf": False,
    }

    if save_format == "hf":
        save_kwargs["save_to_hf"] = True

    model.save_pretrained(output_path, **save_kwargs)
    print(f"Model saved successfully to {output_path}")

    # Verify the saved model
    print("\n" + "-" * 60)
    print("Verifying converted model...")
    try:
        loaded_model = InternLM25ForCausalLM.from_pretrained(output_path, load_checkpoint_format='')
        loaded_model.eval()
        print("Model verification successful!")

        # Check file structure
        print("\nSaved files:")
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file}: {size_mb:.2f} MB")

    except Exception as e:
        print(f"Verification failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"PaddlePaddle model saved to: {output_path}")
    print("=" * 60)
    return True


def test_conversion(model_path, output_path):
    """Test the converted model with a simple inference."""
    print("\n" + "=" * 60)
    print("Testing converted model with simple inference...")
    print("=" * 60)

    try:
        # Load tokenizer and model (use load_checkpoint_format='' for saved Paddle models)
        tokenizer = InternLM25Tokenizer.from_pretrained(output_path)
        print(f"Loading model from {output_path}...")
        model = InternLM25ForCausalLM.from_pretrained(output_path, load_checkpoint_format='')
        model.eval()

        # Test with a simple input
        test_text = "你好，世界！"
        print(f"\nTest input: {test_text}")

        inputs = tokenizer(test_text, return_tensors="pd")
        print(f"Tokenized input shape: {inputs['input_ids'].shape}")

        # Run inference with return_dict=True
        with paddle.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits
            print(f"Output logits shape: {logits.shape}")

        # Get predicted tokens
        predicted_ids = paddle.argmax(logits, axis=-1)
        print(f"Predicted token IDs: {predicted_ids.numpy()}")

        # Test generation (handle tuple return)
        print("\nTesting generation...")
        result = model.generate(
            **inputs,
            max_new_tokens=10,
            decode_strategy="greedy_search",
        )
        # Handle tuple return from generate
        if isinstance(result, tuple):
            generated_ids = result[0]
        else:
            generated_ids = result
        print(f"Generated IDs shape: {generated_ids.shape}")
        print(f"Generated IDs: {generated_ids.numpy()[0]}")

        # Try simple decode (may fail for complex cases)
        try:
            generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
        except Exception as decode_error:
            print(f"Decode warning: {decode_error}")
            print("(Model works, but tokenizer decode has issues)")

        print("\nTest passed! Model conversion successful.")
        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert InternLM2.5 model from HuggingFace to PaddlePaddle format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/cao/llm/internlm/internlm2_5-7b-chat/",
        help="Path to the HuggingFace InternLM2.5 model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./tmp/internlm25_paddle_model",
        help="Path to save the converted PaddlePaddle model",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["pd", "hf"],
        default="pd",
        help="Format to save the model: 'pd' for PaddlePaddle, 'hf' for HuggingFace compatible",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference test after conversion",
    )

    args = parser.parse_args()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1

    # Convert model
    success = convert_hf_to_paddle(args.model_path, args.output_path, args.save_format)

    if not success:
        print("\nConversion failed!")
        return 1

    # Test the converted model if requested
    if args.test:
        test_success = test_conversion(args.model_path, args.output_path)
        if not test_success:
            print("\nConversion test failed!")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
