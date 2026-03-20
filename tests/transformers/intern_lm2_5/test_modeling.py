# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    InternLM25Config,
    InternLM25ForCausalLM,
    InternLM25Tokenizer,
)
from paddleformers.transformers.model_utils import load_sharded_checkpoint
from tests.testing_utils import require_package, slow

class TestInternLM25Config(unittest.TestCase):
    def test_config_initialization(self):
        config = InternLM25Config()
        self.assertEqual(config.vocab_size, 103168)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)

    def test_config_custom_values(self):
        config = InternLM25Config(
            vocab_size=10000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
        )
        self.assertEqual(config.vocab_size, 10000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 14336)

    def test_config_save_and_load(self):
        config = InternLM25Config(vocab_size=10000, hidden_size=4096)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = InternLM25Config.from_pretrained(temp_dir)
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


class InternLM25ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=False,
        )

    def test_model_initialization(self):
        model = InternLM25ForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 256)

    def test_model_forward(self):
        model = InternLM25ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_generation(self):
        model = InternLM25ForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, self.config.vocab_size, [1, 5])

        with paddle.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=20,
                min_length=10,
                use_cache=False,
            )

        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        self.assertGreaterEqual(generated_ids.shape[1], 10)
        self.assertLessEqual(generated_ids.shape[1], 20)

    def test_model_save_and_load(self):
        model = InternLM25ForCausalLM(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf=False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM25ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    def test_chat_method(self):
        """Test the chat method."""
        model = InternLM25ForCausalLM(self.config)
        model.eval()

        # This is a basic test to ensure the method exists and is callable
        # Full testing would require a tokenizer
        self.assertTrue(hasattr(model, "chat"))
        self.assertTrue(hasattr(model, "build_inputs"))
        self.assertTrue(hasattr(model, "stream_chat"))

    def test_model_with_attention_mask(self):
        model = InternLM25ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])
        attention_mask = paddle.ones([batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_with_past_key_values(self):
        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=True,
        )
        model = InternLM25ForCausalLM(config)
        model.eval()

        batch_size = 1
        seq_length = 5
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            # First forward pass
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values

            # Second forward pass with past_key_values
            next_input_ids = paddle.randint(0, config.vocab_size, [batch_size, 1])
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        self.assertIsNotNone(outputs.past_key_values)


class InternLM25ForSequenceClassificationTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            num_labels=3,
        )

    def test_classification_model_initialization(self):
        from paddleformers.transformers import InternLM25ForSequenceClassification

        model = InternLM25ForSequenceClassification(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_labels, 3)

    def test_classification_model_forward(self):
        from paddleformers.transformers import InternLM25ForSequenceClassification

        model = InternLM25ForSequenceClassification(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, self.config.num_labels])


class InternLM25ForQuestionAnsweringTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
        )

    def test_qa_model_initialization(self):
        from paddleformers.transformers import InternLM25ForQuestionAnswering

        model = InternLM25ForQuestionAnswering(self.config)
        self.assertIsNotNone(model)

    def test_qa_model_forward(self):
        from paddleformers.transformers import InternLM25ForQuestionAnswering

        model = InternLM25ForQuestionAnswering(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        self.assertEqual(start_logits.shape, [batch_size, seq_length])
        self.assertEqual(end_logits.shape, [batch_size, seq_length])


class InternLM25ForTokenClassificationTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            num_labels=5,
        )

    def test_token_classification_model_initialization(self):
        from paddleformers.transformers import InternLM25ForTokenClassification

        model = InternLM25ForTokenClassification(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_labels, 5)

    def test_token_classification_model_forward(self):
        from paddleformers.transformers import InternLM25ForTokenClassification

        model = InternLM25ForTokenClassification(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.num_labels])


class InternLM25CompatibilityTest(unittest.TestCase):
    """Test compatibility with PyTorch model conversion."""

    tiny_torch_model_path = "path/to/internlm2.5/tiny_raw"
    tiny_torch_model = None

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        # Skip if model path is not available
        try:
            from transformers import AutoModelForCausalLM

            cls.tiny_torch_model = AutoModelForCausalLM.from_pretrained(
                cls.tiny_torch_model_path, trust_remote_code=True
            )
            cls.tiny_torch_model.eval()
        except Exception:
            cls.tiny_torch_model = None

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    @require_package("transformers", "torch")
    def test_intern_converter(self):
        if self.tiny_torch_model is None:
            self.skipTest("Tiny torch model not available")

        input_ids = np.random.randint(100, 200, [1, 20])
        paddle_model = InternLM25ForCausalLM.from_pretrained(
            self.tiny_torch_model_path, convert_from_hf=True, load_checkpoint_format=""
        )
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        import torch

        torch_logit = self.tiny_torch_model(torch.tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                rtol=1e-5,
            )
        )


class InternLM25ConvertedWeightTest(unittest.TestCase):
    """Test the converted weights from HuggingFace model."""

    def setUp(self):
        # Get the project root directory and construct the model path
        import os
        # From tests/transformers/intern_lm2_5/test_modeling.py go to project root (4 levels up)
        # tests/transformers/intern_lm2_5/test_modeling.py -> tests/transformers/intern_lm2_5 -> tests/transformers -> tests -> project_root
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        # Construct model path
        self.model_path = os.path.expanduser("~/llm/internlm/internlm2_5-1_8b-chat-paddle")
        self.hf_model_path = os.path.expanduser("~/llm/internlm/internlm2_5-1_8b-chat-raw")

        # Save original dtype and set to bfloat16 for tests
        self._original_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("bfloat16")

    def tearDown(self):
        # Restore original dtype
        paddle.set_default_dtype(self._original_dtype)

    def test_model_loading(self):
        """Test loading the converted model."""
        model = InternLM25ForCausalLM.from_pretrained(self.model_path, load_checkpoint_format="")
        self.assertIsNotNone(model)
        model.eval()

    def test_model_inference(self):
        """Test basic inference with the converted model."""
        model = InternLM25ForCausalLM.from_pretrained(self.model_path, load_checkpoint_format="")
        model.eval()

        # Prepare input
        input_ids = paddle.to_tensor([[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588]])
        attention_mask = paddle.to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)  # batch_size
        self.assertEqual(logits.shape[1], 10)  # seq_length
        self.assertIsNotNone(logits)

    def test_model_generation(self):
        """Test text generation with the converted model."""
        model = InternLM25ForCausalLM.from_pretrained(self.model_path, load_checkpoint_format="")
        tokenizer = InternLM25Tokenizer.from_pretrained(self.model_path, load_checkpoint_format="")
        model.eval()

        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pd")
        input_ids = inputs["input_ids"]

        with paddle.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                use_cache=True,
                decode_strategy="greedy_search",
            )

        # Extract from tuple if needed
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        # Verify generation worked
        self.assertIsNotNone(generated_ids)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        # Decode the output
        decoded = tokenizer.decode(generated_ids[0].numpy().tolist(), skip_special_tokens=True)
        self.assertGreater(len(decoded.strip()), 0)

    #@slow
    def test_chinese_generation(self):
        """Test Chinese text generation with the converted model using chat mode."""
        paddle.set_device("gpu")

        config = InternLM25Config.from_pretrained(self.model_path)
        model = InternLM25ForCausalLM(config)
        
        load_sharded_checkpoint(model, self.model_path, strict=False, prefer_safe=False)
        model.eval()

        tokenizer = InternLM25Tokenizer.from_pretrained(self.model_path, load_checkpoint_format="")

        prompt = "猫和狗的区别是什么，列出主要的3点"
        meta_instruction = "You are a helpful assistant. Please answer in plain text without markdown."
        chat_inputs = model.build_inputs(
            tokenizer, prompt, history=[], meta_instruction=meta_instruction
        )

        with paddle.no_grad():
            out = model.generate(
                input_ids=chat_inputs["input_ids"],
                attention_mask=chat_inputs.get("attention_mask"),
                max_new_tokens=128,
                use_cache=True,
                decode_strategy="greedy_search",
            )

        seq = out[0] if isinstance(out, (list, tuple)) else out

        # Decode the output
        decoded = tokenizer.decode(seq.numpy().tolist()[0], skip_special_tokens=True)

        # Print the result for observation
        print("\n" + "=" * 80)
        print("Chinese Generation Test (Chat Mode)")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Generated: {decoded}")
        print("=" * 80 + "\n")

        self.assertGreater(len(decoded.strip()), 0)

    @slow
    def test_hf_direct_load_and_inference(self):
        """Test direct loading from raw HuggingFace-format InternLM2.5 checkpoint and run Chinese inference."""
        if not os.path.isdir(self.hf_model_path):
            self.skipTest(f"HF model directory not found: {self.hf_model_path}")
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required for this test")

        paddle.set_device("gpu")
        paddle.set_default_dtype("bfloat16")

        model = InternLM25ForCausalLM.from_pretrained(
            self.hf_model_path,
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
        )
        model.eval()
        tokenizer = InternLM25Tokenizer.from_pretrained(self.hf_model_path, load_checkpoint_format="")

        prompt = "猫和狗的区别是什么，列出主要的3点"
        meta_instruction = "You are a helpful assistant. Please answer in plain text without markdown."
        inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction=meta_instruction)
        with paddle.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=128,
                use_cache=True,
                decode_strategy="greedy_search",
            )

        generated_ids = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        decoded = tokenizer.decode(generated_ids[0].numpy().tolist(), skip_special_tokens=True)
        print("\n[HF Direct Load] prompt:", prompt)
        print("[HF Direct Load] response:", decoded)

        self.assertIsNotNone(decoded)
        self.assertGreater(len(decoded.strip()), 0)


if __name__ == "__main__":
    unittest.main()
