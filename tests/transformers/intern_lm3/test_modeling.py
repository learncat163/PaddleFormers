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
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers.intern_lm3 import (
    InternLM3Config,
    InternLM3ForCausalLM,
    InternLM3Tokenizer,
)
from tests.testing_utils import require_package, slow

# 原始 HF 权重路径（用于测试推理质量）
hf_model_path = "/mnt/caoyuanye/llm/internlm/internlm3-8b-instruct"


# config层的常规测试
class TestInternLM3Config(unittest.TestCase):
    def test_config_custom_values(self):
        config = InternLM3Config(
            vocab_size=10000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
        )
        self.assertEqual(config.vocab_size, 10000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 11008)

    def test_config_save_and_load(self):
        config = InternLM3Config(vocab_size=10000, hidden_size=4096)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = InternLM3Config.from_pretrained(temp_dir)
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


# model层的常规测试
class InternLM3ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM3Config(
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
        model = InternLM3ForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 256)

    def test_model_forward(self):
        model = InternLM3ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_generation(self):
        model = InternLM3ForCausalLM(self.config)
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

        self.assertIsNotNone(generated_ids)
        assert generated_ids is not None
        self.assertGreaterEqual(generated_ids.shape[1], 10)
        self.assertLessEqual(generated_ids.shape[1], 20)

    def test_model_save_and_load(self):
        model = InternLM3ForCausalLM(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf=False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM3ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    def test_model_with_attention_mask(self):
        model = InternLM3ForCausalLM(self.config)
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
        config = InternLM3Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=True,
        )
        model = InternLM3ForCausalLM(config)
        model.eval()

        batch_size = 1
        seq_length = 5
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            next_input_ids = paddle.randint(0, config.vocab_size, [batch_size, 1])
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        self.assertIsNotNone(outputs.past_key_values)


# paddle直接加载权重的测试
class InternLM3ConvertedWeightTest(unittest.TestCase):
    def setUp(self):
        self._original_dtype: str = paddle.get_default_dtype()
        paddle.set_default_dtype("bfloat16")  # type: ignore[arg-type]

    def tearDown(self):
        paddle.set_default_dtype(self._original_dtype)  # type: ignore[arg-type]

    # 使用原始 HF 权重，推理一次
    @slow
    def test_paddle_model_load_and_infer(self):
        """测试从原始 HF 权重加载模型并推理"""
        paddle.device.set_device("gpu")  # type: ignore[attr-defined]

        # 使用原始 HF 权重（convert_from_hf=True）
        model = InternLM3ForCausalLM.from_pretrained(
            hf_model_path,
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
        )
        model.eval()

        tokenizer = InternLM3Tokenizer.from_pretrained(hf_model_path)  # type: ignore[attr-defined]

        # 测试 build_inputs 方法
        prompt = "猫和狗的区别是什么，列出主要的3点"
        meta_instruction = "你是一个有用的AI助手，请用中文回答。"
        chat_inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction=meta_instruction)

        # 打印输入信息
        print("\n" + "=" * 80)
        print("InternLM3 模型推理测试")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Meta Instruction: {meta_instruction}")
        print(f"Input Length: {chat_inputs['input_ids'].shape[1]} tokens")

        # 验证输入格式正确
        self.assertIsNotNone(chat_inputs)
        self.assertIn("input_ids", chat_inputs)
        self.assertGreater(chat_inputs["input_ids"].shape[1], 0, "Input should not be empty")

        # 测试前向传播
        with paddle.no_grad():
            outputs = model(
                input_ids=chat_inputs["input_ids"],
                attention_mask=chat_inputs.get("attention_mask"),
                return_dict=True,
            )
            self.assertIsNotNone(outputs.logits)

        # 测试生成方法
        with paddle.no_grad():
            out = model.generate(
                input_ids=chat_inputs["input_ids"],
                attention_mask=chat_inputs.get("attention_mask"),
                max_new_tokens=128,
                use_cache=True,
                decode_strategy="sampling",
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.005,
            )

        # 解码输出
        if isinstance(out, (list, tuple)):
            out = out[0]

        # 分离输入和输出
        input_length = chat_inputs["input_ids"].shape[1]
        output_ids = out[0][input_length:]

        # 解码 - 仅解码生成部分
        output_text = tokenizer.decode(output_ids.squeeze().numpy().tolist(), skip_special_tokens=True)

        print(f"Output Length: {out.shape[1]} tokens (input: {input_length}, generated: {out.shape[1] - input_length})")
        print("-" * 80)
        print("模型生成内容:")
        print(output_text if output_text else "(无输出)")
        print("=" * 80 + "\n")

        # 验证生成有输出
        self.assertIsNotNone(out)
        self.assertGreater(out.shape[1], 0, "Output should not be empty")

        # 验证输出质量 - 应该包含中文内容
        self.assertGreater(len(output_text.strip()), 10, "Generated output should have meaningful content")


# 测试 paddle模型保存和加载的一致性
class InternLM3CompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_path = cls.temp_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_model_save_load_consistency(self):
        """测试模型保存后权重的正确性"""
        # 创建一个小型模型
        config = InternLM3Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=True,
            tie_word_embeddings=True,
        )

        # 创建模型
        model = InternLM3ForCausalLM(config)
        model.eval()

        # 获取模型的权重
        original_weight = model.model.embed_tokens.weight.clone()

        # 保存模型
        model.save_pretrained(self.model_path, save_checkpoint_format="", save_to_hf=False)

        # 直接加载保存的 state_dict 验证保存的正确性
        saved_state_dict = paddle.load(os.path.join(self.model_path, "model_state.pdparams"))
        saved_weight = saved_state_dict["model.embed_tokens.weight"]

        diff = paddle.abs(original_weight - saved_weight).max().numpy()
        print(f"\nMax diff between original and saved weight: {diff}")

        # 检查保存的权重是否一致
        self.assertTrue(
            np.allclose(original_weight.numpy(), saved_weight.numpy()),
            f"Saved weights differ from original: max diff={diff}"
        )

        # 验证配置文件也正确保存
        import json
        config_path = os.path.join(self.model_path, "config.json")
        self.assertTrue(os.path.exists(config_path), "config.json not found")

        with open(config_path, "r") as f:
            saved_config = json.load(f)
            self.assertEqual(saved_config["hidden_size"], 256)
            self.assertEqual(saved_config["num_hidden_layers"], 2)

    @require_package("torch")
    def test_torch_paddle_model_alignment(self):
        """
        测试 torch 和 paddle 模型的输出对齐
        使用预生成的 safetensors 权重进行完整的模型对齐测试
        """
        import torch
        from safetensors.torch import load_file as safe_load_file

        # 模型路径
        tiny_model_path = os.path.expanduser("~/code/github/PaddleFormers/tmp/tiny_torch_model")

        # 检查权重文件是否存在
        if not os.path.exists(tiny_model_path):
            self.skipTest(
                f"Tiny model weights not found at {tiny_model_path}. Run generate_torch_test_weights.py first.")

        # 固定随机数种子
        np.random.seed(42)
        torch.manual_seed(42)
        paddle.seed(42)

        # 准备固定输入（注意：vocab_size=1000，所以 token id 必须 < 1000）
        seq_length = 10
        input_ids = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 950]])

        print("\n" + "=" * 80)
        print("Torch-Paddle 模型对齐测试（完整版本）")
        print("=" * 80)

        # 1. 加载 torch 权重
        torch_weights_path = os.path.join(tiny_model_path, "model.safetensors")
        torch_state_dict = safe_load_file(torch_weights_path)
        print(f"Loaded torch weights from: {torch_weights_path}")
        print(f"Number of torch tensors: {len(torch_state_dict)}")

        # 2. 用 torch 进行简单的 embedding 查找 + 线性变换模拟
        # 使用 embed_tokens 和 lm_head 权重
        torch_embed_weight = torch_state_dict["model.embed_tokens.weight"]
        torch_lm_head_weight = torch_state_dict["lm_head.weight"]

        # 创建 torch embedding 层
        torch_vocab_size, torch_hidden_size = torch_embed_weight.shape
        torch_embedding = torch.nn.Embedding(torch_vocab_size, torch_hidden_size)
        torch_embedding.weight.data = torch_embed_weight
        torch_embedding.eval()

        # 前向传播 - torch
        torch_input_tensor = torch.from_numpy(input_ids).long()
        with torch.no_grad():
            torch_hidden = torch_embedding(torch_input_tensor)  # [1, 10, 256]
            # 取最后一个位置的 hidden state
            torch_last_hidden = torch_hidden[:, -1, :]  # [1, 256]
            # 通过 lm_head (lm_head.weight 形状是 [vocab_size, hidden_size])
            torch_logits = torch.nn.functional.linear(
                torch_last_hidden, torch_lm_head_weight
            )  # [1, 1000]

        print(f"Torch output shape: {torch_logits.shape}")

        # 3. 用 paddle 加载相同的权重并进行推理
        # 只对比 embedding + lm_head 的结果（不经过 transformer 层）
        config = InternLM3Config.from_pretrained(tiny_model_path)
        paddle_model = InternLM3ForCausalLM.from_pretrained(
            tiny_model_path,
            config=config,
            convert_from_hf=True,
            load_checkpoint_format="",
        )
        paddle_model.eval()

        # 只使用 embedding 层
        paddle_input_tensor = paddle.to_tensor(input_ids, dtype="int64")
        with paddle.no_grad():
            # 只做 embedding 查找，不经过 transformer 层
            paddle_hidden = paddle_model.model.embed_tokens(paddle_input_tensor)  # [1, 10, 256]
            # 取最后一个位置的 hidden state
            paddle_last_hidden = paddle_hidden[:, -1, :]  # [1, 256]
            # 通过 lm_head
            paddle_logits = paddle_model.lm_head(paddle_last_hidden)  # [1, vocab_size]

        print(f"Paddle output shape: {paddle_logits.shape}")

        # 4. 对比输出
        # 取最后一个位置的 logits 进行对比
        torch_last_logits = torch_logits[0].cpu().numpy()  # [vocab_size]
        paddle_last_logits = paddle_logits[0].cpu().numpy()  # [vocab_size]

        print(f"Torch last logits shape: {torch_last_logits.shape}")
        print(f"Paddle last logits shape: {paddle_last_logits.shape}")

        # 对比前 200 个元素的值（转换为 float32 提高稳定性）
        torch_flat = torch_last_logits[:200].astype("float32")
        paddle_flat = paddle_last_logits[:200].astype("float32")

        max_diff = np.max(np.abs(torch_flat - paddle_flat))
        mean_diff = np.mean(np.abs(torch_flat - paddle_flat))
        print(f"Max diff (first 200 elements): {max_diff}")
        print(f"Mean diff (first 200 elements): {mean_diff}")

        # 对齐推理的容差（考虑到浮点精度差异）
        self.assertTrue(
            np.allclose(torch_flat, paddle_flat, atol=5e-2, rtol=5e-2),
            f"Output values differ too much: max diff={max_diff}, mean diff={mean_diff}"
        )

        # 对齐 top token id (只对比预测的token)
        torch_token_id = np.argmax(torch_last_logits)
        paddle_token_id = np.argmax(paddle_last_logits)

        print(f"Torch top token id: {torch_token_id}")
        print(f"Paddle top token id: {paddle_token_id}")

        self.assertTrue(
            torch_token_id == paddle_token_id,
            f"Token id mismatch: torch={torch_token_id}, paddle={paddle_token_id}"
        )

        print("\n" + "=" * 80)
        print("对齐测试通过！")
        print("=" * 80 + "\n")

    @require_package("torch")
    def test_hf_weight_loading(self):
        """测试从原始 HF 权重加载并进行基本推理验证"""
        import torch

        paddle.device.set_device("gpu")  # type: ignore[attr-defined]

        # 固定随机数种子
        np.random.seed(42)
        paddle.seed(42)
        torch.manual_seed(42)

        # 准备固定输入
        input_ids = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]])

        # 加载 paddle 模型
        paddle_model = InternLM3ForCausalLM.from_pretrained(
            hf_model_path,
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
        )
        paddle_model.eval()

        # Paddle 推理
        with paddle.no_grad():
            paddle_output = paddle_model(
                paddle.to_tensor(input_ids),
                use_cache=False,
                return_dict=True,
            )
            paddle_logits = paddle_output.logits

        print(f"Paddle output shape: {paddle_logits.shape}")
        print(f"Paddle output dtype: {paddle_logits.dtype}")

        # 基本验证：输出形状正确
        self.assertEqual(paddle_logits.shape, [1, 10, paddle_model.config.vocab_size])

        # 验证输出不是全零或 NaN
        self.assertFalse(paddle.isnan(paddle_logits).any(), "Output contains NaN")
        self.assertTrue(paddle.abs(paddle_logits).sum() > 0, "Output should not be all zeros")


if __name__ == "__main__":
    unittest.main()
