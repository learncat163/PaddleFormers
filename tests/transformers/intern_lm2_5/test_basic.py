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
"""Basic test for InternLM2.5 model save/load/inference."""

import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import paddle


def test_basic_import():
    """Test basic import of InternLM2.5 components."""
    print("=" * 50)
    print("Test 1: Basic Import")
    print("=" * 50)

    try:
        from paddleformers.transformers import InternLM25Config
        print("InternLM25Config imported successfully")
    except Exception as e:
        print(f"Failed to import InternLM25Config: {e}")
        return False

    try:
        from paddleformers.transformers import InternLM25ForCausalLM
        print("InternLM25ForCausalLM imported successfully")
    except Exception as e:
        print(f"Failed to import InternLM25ForCausalLM: {e}")
        return False

    print("Import test PASSED")
    return True


def test_config_creation():
    """Test config creation."""
    print("\n" + "=" * 50)
    print("Test 2: Config Creation")
    print("=" * 50)

    try:
        from paddleformers.transformers import InternLM25Config

        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
        )
        print(f"Config created: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        print("Config creation test PASSED")
        return True
    except Exception as e:
        print(f"Failed to create config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\n" + "=" * 50)
    print("Test 3: Model Creation")
    print("=" * 50)

    try:
        from paddleformers.transformers import InternLM25Config, InternLM25ForCausalLM

        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
        )

        model = InternLM25ForCausalLM(config)
        print(f"Model created successfully")
        print(f"Model type: {type(model)}")
        print("Model creation test PASSED")
        return True
    except Exception as e:
        print(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 50)
    print("Test 4: Model Forward Pass")
    print("=" * 50)

    try:
        from paddleformers.transformers import InternLM25Config, InternLM25ForCausalLM

        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
        )

        model = InternLM25ForCausalLM(config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Expected shape: [{batch_size}, {seq_length}, {config.vocab_size}]")

        assert logits.shape == [batch_size, seq_length, config.vocab_size], \
            f"Shape mismatch: {logits.shape} vs [{batch_size}, {seq_length}, {config.vocab_size}]"

        print("Model forward pass test PASSED")
        return True
    except Exception as e:
        print(f"Failed model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """Test model save and load."""
    print("\n" + "=" * 50)
    print("Test 5: Model Save and Load")
    print("=" * 50)

    tmp_dir = None
    try:
        from paddleformers.transformers import InternLM25Config, InternLM25ForCausalLM

        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
        )

        # Create and save model
        model = InternLM25ForCausalLM(config)
        model.eval()

        tmp_dir = tempfile.mkdtemp(prefix="internlm25_test_")
        print(f"Saving model to: {tmp_dir}")

        model.save_pretrained(tmp_dir, save_checkpoint_format="", save_to_hf=False)

        # Check files exist
        model_file = os.path.join(tmp_dir, "model_state.pdparams")
        config_file = os.path.join(tmp_dir, "config.json")

        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            return False
        if not os.path.exists(config_file):
            print(f"Config file not found: {config_file}")
            return False

        print(f"Model files saved: {os.listdir(tmp_dir)}")

        # Load model
        print("Loading model...")
        loaded_model = InternLM25ForCausalLM.from_pretrained(tmp_dir, load_checkpoint_format="")
        loaded_model.eval()

        # Test loaded model
        input_ids = paddle.randint(0, config.vocab_size, [1, 5])

        with paddle.no_grad():
            outputs = loaded_model(input_ids=input_ids, return_dict=True)

        print(f"Loaded model output shape: {outputs.logits.shape}")

        print("Model save/load test PASSED")
        return True
    except Exception as e:
        print(f"Failed model save/load: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if tmp_dir and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temp directory: {tmp_dir}")


def test_model_generation():
    """Test model generation."""
    print("\n" + "=" * 50)
    print("Test 6: Model Generation")
    print("=" * 50)

    try:
        from paddleformers.transformers import InternLM25Config, InternLM25ForCausalLM

        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
        )

        model = InternLM25ForCausalLM(config)
        model.eval()

        input_ids = paddle.randint(0, config.vocab_size, [1, 5])

        print(f"Input shape: {input_ids.shape}")

        with paddle.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=15,
                use_cache=False,
            )

        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        print(f"Generated shape: {generated_ids.shape}")
        print(f"Generated length: {generated_ids.shape[1]}")

        print("Model generation test PASSED")
        return True
    except Exception as e:
        print(f"Failed model generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("=" * 50)
    print("InternLM2.5 Basic Functionality Test Suite")
    print("=" * 50)

    tests = [
        ("Basic Import", test_basic_import),
        ("Config Creation", test_config_creation),
        ("Model Creation", test_model_creation),
        ("Model Forward Pass", test_model_forward),
        ("Model Save/Load", test_model_save_load),
        ("Model Generation", test_model_generation),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nTest '{name}' raised unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print("=" * 50)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
