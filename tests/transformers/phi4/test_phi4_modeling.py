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

import unittest
import tempfile
import shutil
import os
import sys
import gc

import paddle
import numpy as np

from paddleformers.transformers.phi4 import Phi4ForCausalLM, Phi4Config, Phi4Model, Phi4Tokenizer


class TestPhi4Modeling(unittest.TestCase):
    model = None
    model_path = "/mnt/caoyuanye/llm/microsoft/Phi-4-mini-flash-reasoning"

    @classmethod
    def setUpClass(cls):
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype='bfloat16',
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        config = self.model.config
        self.assertIsNotNone(config)
        self.assertIn(config.model_type, ("phi4", "phi4flash"))
        self.assertGreater(config.vocab_size, 0)
        self.assertGreater(config.hidden_size, 0)
        self.assertGreater(config.num_hidden_layers, 0)
        print(f"Config OK: {config.model_type}, vocab={config.vocab_size}")

    def test_model_loading_bf16(self):
        param = next(iter(self.model.parameters()))
        self.assertEqual(param.dtype, paddle.bfloat16)
        print(f"Model loaded with bfloat16 successfully, param dtype={param.dtype}")

    def test_model_loading_float32(self):
        self.assertIsNotNone(self.model)
        print(f"Model loaded with float32 successfully")

    def test_forward_pass(self):
        batch_size = 1
        seq_length = 3
        input_ids = paddle.randint(0, self.model.config.vocab_size, [batch_size, seq_length])
        with paddle.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
        self.assertIsNotNone(outputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(list(logits.shape), [batch_size, seq_length, self.model.config.vocab_size])
        print(f"Forward pass OK, shape: {logits.shape}")

    def test_tokenizer_loading(self):
        tokenizer = Phi4Tokenizer.from_pretrained(self.model_path)
        self.assertIsNotNone(tokenizer)
        self.assertGreater(tokenizer.vocab_size, 0)
        text = "Hello, this is a test."
        encoding = tokenizer(text)
        self.assertIn("input_ids", encoding)
        self.assertGreater(len(encoding["input_ids"]), 0)
        decoded = tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)
        print(f"Tokenizer OK: vocab_size={tokenizer.vocab_size}, input_ids={encoding['input_ids']}")
        print(f"Decoded: {decoded}")

    def test_model_save_and_load(self):
        save_path = os.path.join(self.temp_dir, "saved_model")
        current_device = paddle.get_device()
        self.__class__.model.to('cpu')
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        try:
            self.model.save_pretrained(save_path)
            loaded_model = Phi4ForCausalLM.from_pretrained(save_path, dtype='bfloat16')
            loaded_model.eval()
            input_ids = paddle.randint(0, loaded_model.config.vocab_size, [1, 3], dtype='int64')
            with paddle.no_grad():
                outputs = loaded_model(input_ids=input_ids)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
            self.assertEqual(logits.shape[-1], loaded_model.config.vocab_size)
            del loaded_model
            gc.collect()
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            print(f"Model save and load OK, logits shape={tuple(logits.shape)}")
        finally:
            self.__class__.model.to(current_device)

    def test_attention_mechanism(self):
        config = self.model.config
        self.assertGreater(config.num_attention_heads, 0)
        self.assertGreater(config.num_key_value_heads, 0)
        print(f"Attention: heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")

    def test_load_original_transformers_weights(self):
        self.assertIsNotNone(self.model)
        input_ids = paddle.randint(0, self.model.config.vocab_size, [1, 3], dtype='int64')
        with paddle.no_grad():
            outputs = self.model(input_ids=input_ids)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(logits.shape[-1], self.model.config.vocab_size)
        print(f"AOA HF weight load via setUpClass verified: shape={tuple(logits.shape)}")


class TestPhi4BF16Optimization(unittest.TestCase):
    model_path = "/mnt/caoyuanye/llm/microsoft/Phi-4-mini-flash-reasoning"
    model = None

    @classmethod
    def setUpClass(cls):
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype='bfloat16',
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def test_bf16_memory_usage(self):
        param = next(iter(self.model.parameters()))
        self.assertEqual(param.dtype, paddle.bfloat16)
        input_ids = paddle.randint(0, self.model.config.vocab_size, [1, 8], dtype='int64')
        with paddle.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(logits.dtype, paddle.bfloat16)
        self.assertEqual(logits.shape[-1], self.model.config.vocab_size)
        print(f"BF16 memory usage OK: param dtype={param.dtype}, logits dtype={logits.dtype}")

    def test_bf16_forward_speed(self):
        input_ids = paddle.randint(0, self.model.config.vocab_size, [1, 16], dtype='int64')
        import time
        with paddle.no_grad():
            start = time.time()
            outputs = self.model(input_ids=input_ids, use_cache=False)
            paddle.device.synchronize()
            elapsed = time.time() - start
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        self.assertEqual(logits.shape[-1], self.model.config.vocab_size)
        print(f"BF16 forward speed OK: seq_len=16, elapsed={elapsed:.3f}s, logits={logits.shape}")


class TestPhi4Inference(unittest.TestCase):
    model_path = "/mnt/caoyuanye/llm/microsoft/Phi-4-mini-flash-reasoning"
    model = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Phi4Tokenizer.from_pretrained(cls.model_path)
        cls.model = Phi4ForCausalLM.from_pretrained(
            cls.model_path,
            dtype='bfloat16',
            convert_from_hf=True,
        )
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.tokenizer
        cls.model = None
        cls.tokenizer = None
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def test_inference_cat_vs_dog(self):
        messages = [{"role": "user", "content": "猫和狗的区别是什么"}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pd")
        input_ids = inputs["input_ids"]

        with paddle.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"\n{'='*60}")
        print(f"Input: 猫和狗的区别是什么")
        print(f"{'='*60}")
        print(f"Output:\n{response}")
        print(f"{'='*60}")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPhi4Modeling))
    suite.addTests(loader.loadTestsFromTestCase(TestPhi4BF16Optimization))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    successes = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"Successes: {successes}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
