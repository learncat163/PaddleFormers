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


import os
import unittest

import numpy as np
import paddle

from tests.testing_utils import require_package, slow

_MODEL_HF_ID = "apple/OpenELM-1_1B-Instruct"
# OpenELM 不自带 tokenizer，使用 LLaMA tokenizer（与 OpenELM 原始代码一致）
_TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"
_CONVERTED_WEIGHT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    ".claude",
    "ai_history",
    "openelm",
    "2-weight",
    "convert-weight",
)
_PROMPT_INFERENCE = "What is the difference between cats and dogs?"
_SEED = 42

# OpenELM 小模型配置，用于快速单元测试
# 注意: OpenELM 使用 model_dim / num_transformer_layers 而非 hidden_size / num_hidden_layers
SMALL_CONFIG = {
    "vocab_size": 1000,
    "max_context_length": 128,
    "num_transformer_layers": 2,
    "model_dim": 64,
    "head_dim": 16,
    "qkv_multipliers": 1.0,
    "num_gqa_groups": 1,
    "ffn_multipliers": 2.0,
    "ffn_with_glu": True,
    "ffn_dim_divisor": 64,
    "activation_fn_name": "swish",
    "normalize_qk_projections": False,
    "share_input_output_layers": True,
    "rope_freq_constant": 10000,
    "rope_max_length": 256,
    "initializer_range": 0.02,
    "use_cache": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


class TestOpenELMModel(unittest.TestCase):
    """OpenELM 基本模型测试 (OpenELMModel, base model without LM head)"""

    BATCH = 2
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMModel

        self.config = OpenELMConfig(**SMALL_CONFIG)
        self.model = OpenELMModel(self.config)
        self.model.eval()

    def test_model_from_config(self):
        """测试从配置创建模型"""
        from paddleformers.transformers import OpenELMConfig, OpenELMModel

        config = OpenELMConfig(**SMALL_CONFIG)
        model = OpenELMModel(config)
        self.assertIsNotNone(model)

        # 验证配置属性
        self.assertEqual(config.vocab_size, SMALL_CONFIG["vocab_size"])
        self.assertEqual(config.model_dim, SMALL_CONFIG["model_dim"])
        self.assertEqual(config.num_transformer_layers, SMALL_CONFIG["num_transformer_layers"])
        self.assertEqual(config.head_dim, SMALL_CONFIG["head_dim"])

        # num_hidden_layers 是 num_transformer_layers 的别名
        self.assertEqual(config.num_hidden_layers, config.num_transformer_layers)

        # num_query_heads 和 num_kv_heads 应该是列表
        self.assertIsInstance(config.num_query_heads, list)
        self.assertIsInstance(config.num_kv_heads, list)
        self.assertEqual(len(config.num_query_heads), SMALL_CONFIG["num_transformer_layers"])
        self.assertEqual(len(config.num_kv_heads), SMALL_CONFIG["num_transformer_layers"])

    def test_model_forward(self):
        """测试模型前向推理"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)

        # 验证输出是 BaseModelOutputWithPast
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_with_attention_mask(self):
        """测试带注意力掩码的前向推理"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        attn_mask = paddle.ones([self.BATCH, self.SEQ], dtype=paddle.int64)
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)

        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_with_cache(self):
        """测试带 KV cache 的前向推理"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, use_cache=True, return_dict=True)

        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertIsNotNone(output.past_key_values)

    def test_forward_without_cache(self):
        """测试不带 KV cache 的前向推理"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, use_cache=False, return_dict=True)

        self.assertIsNone(output.past_key_values)

    def test_deterministic_output(self):
        """测试确定性输出 (相同输入应产生相同输出)"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 6])
        with paddle.no_grad():
            out1 = self.model(input_ids=input_ids, return_dict=True).last_hidden_state
            out2 = self.model(input_ids=input_ids, return_dict=True).last_hidden_state
        self.assertTrue(paddle.allclose(out1, out2))

    def test_output_dtype_consistent(self):
        """测试输出数据类型一致性"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)
        self.assertTrue(output.last_hidden_state.dtype in [paddle.float32, paddle.float16, paddle.bfloat16])

    def test_forward_with_inputs_embeds(self):
        """测试使用 inputs_embeds 的前向推理"""
        inputs_embeds = paddle.randn([self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]])
        with paddle.no_grad():
            output = self.model(inputs_embeds=inputs_embeds, return_dict=True)

        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_output_hidden_states(self):
        """测试输出隐藏层状态"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

        self.assertIsNotNone(output.hidden_states)
        # num_transformer_layers + 1 (embedding output + each layer output)
        self.assertEqual(
            len(output.hidden_states),
            SMALL_CONFIG["num_transformer_layers"] + 1,
        )

    def test_forward_tuple_output(self):
        """测试 tuple 输出格式 (return_dict=False)"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=False)

        self.assertIsInstance(output, tuple)
        # 第一个元素是 hidden_states
        self.assertEqual(list(output[0].shape), [1, self.SEQ, SMALL_CONFIG["model_dim"]])


class TestOpenELMForCausalLM(unittest.TestCase):
    """OpenELM Causal LM 模型测试"""

    BATCH = 2
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        self.config = OpenELMConfig(**SMALL_CONFIG)
        self.model = OpenELMForCausalLM(self.config)
        self.model.eval()

    def test_causal_lm_from_config(self):
        """测试从配置创建 CausalLM 模型"""
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        config = OpenELMConfig(**SMALL_CONFIG)
        model = OpenELMForCausalLM(config)
        self.assertIsNotNone(model)

        # share_input_output_layers=True 时 lm_head 应为 None
        if config.share_input_output_layers:
            self.assertIsNone(model.lm_head)

    def test_causal_lm_forward(self):
        """测试 CausalLM 前向推理"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)

        # 验证输出是 CausalLMOutputWithPast
        self.assertTrue(hasattr(output, "logits"))
        self.assertEqual(
            list(output.logits.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["vocab_size"]],
        )

    def test_causal_lm_forward_with_labels(self):
        """测试带 labels 的 CausalLM 前向推理 (计算 loss)

        注意: 在 CPU 上运行，因为 Paddle GPU 的 CrossEntropyLoss 不支持 float32 labels
        (模型代码中 shift_labels.cast(shift_logits.dtype) 将 labels 转为 float32)
        """
        original_device = paddle.get_device()
        paddle.set_device("cpu")
        try:
            # 在 CPU 上重新创建模型以避免设备不匹配
            from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

            config = OpenELMConfig(**SMALL_CONFIG)
            model = OpenELMForCausalLM(config)
            model.eval()

            input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
            labels = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
            with paddle.no_grad():
                output = model(input_ids=input_ids, labels=labels, return_dict=True)

            self.assertIsNotNone(output.loss)
            self.assertEqual(output.loss.shape, [])
            self.assertTrue(float(output.loss) > 0)
        finally:
            paddle.set_device(original_device)

    def test_causal_lm_generate_autoregressive(self):
        """测试自回归生成 (手动 step-by-step)"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 5])
        gen_ids = []
        max_gen_tokens = 5

        cur_ids = input_ids.numpy()
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = self.model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)

        self.assertEqual(len(gen_ids), max_gen_tokens)
        # 验证生成的 token 在合法范围内
        for tid in gen_ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, SMALL_CONFIG["vocab_size"])

    def test_causal_lm_deterministic_output(self):
        """测试 CausalLM 确定性输出"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 6])
        with paddle.no_grad():
            out1 = self.model(input_ids=input_ids, return_dict=True).logits
            out2 = self.model(input_ids=input_ids, return_dict=True).logits
        self.assertTrue(paddle.allclose(out1, out2))

    def test_causal_lm_prepare_inputs_for_generation(self):
        """测试 prepare_inputs_for_generation 方法"""
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        attention_mask = paddle.ones([1, 5], dtype="int64")

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            use_cache=True,
        )

        self.assertIn("input_ids", model_inputs)
        self.assertIn("attention_mask", model_inputs)
        self.assertIn("position_ids", model_inputs)
        self.assertIn("use_cache", model_inputs)

    def test_causal_lm_with_cache_autoregressive(self):
        """测试带 KV cache 的自回归生成"""
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 3])
        gen_ids = []
        max_gen_tokens = 3

        with paddle.no_grad():
            # 第一步: 使用完整输入
            out = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
            next_token = int(out.logits[0, -1].argmax().item())
            gen_ids.append(next_token)
            past_kv = out.past_key_values

            # 后续步骤: 使用 cache
            for step in range(1, max_gen_tokens):
                next_input = paddle.to_tensor([[next_token]], dtype="int64")
                attention_mask = paddle.ones([1, input_ids.shape[1] + step], dtype="int64")
                out = self.model(
                    input_ids=next_input,
                    past_key_values=past_kv,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                past_kv = out.past_key_values

        self.assertEqual(len(gen_ids), max_gen_tokens)

    def test_causal_lm_not_share_input_output(self):
        """测试不共享输入输出嵌入层的情况"""
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        config_dict = dict(SMALL_CONFIG)
        config_dict["share_input_output_layers"] = False
        config = OpenELMConfig(**config_dict)
        model = OpenELMForCausalLM(config)
        model.eval()

        # share_input_output_layers=False 时 lm_head 不应为 None
        self.assertIsNotNone(model.lm_head)

        input_ids = paddle.randint(0, config.vocab_size, [1, 5])
        with paddle.no_grad():
            output = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(output.logits.shape), [1, 5, config.vocab_size])


class TestOpenELMToken(unittest.TestCase):
    """OpenELM Token 相关测试

    注意: OpenELM 不自带 tokenizer，原始代码使用 LLaMA tokenizer。
    此处使用公开的 LLaMA tokenizer 进行测试。
    """

    @require_package("transformers")
    def test_tokenizer_load(self):
        """测试 tokenizer 加载"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        self.assertIsNotNone(tokenizer)

    @require_package("transformers")
    def test_tokenizer_encode_decode(self):
        """测试 tokenizer 编码解码"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        text = "Hello, how are you today?"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(decoded, text)

    @require_package("transformers")
    def test_tokenizer_batch_encode(self):
        """测试 tokenizer 批量编码"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        # LLaMA tokenizer 默认没有 pad_token，需要设置
        tokenizer.pad_token = tokenizer.eos_token

        texts = [
            "Hello, how are you today?",
            "What is the difference between cats and dogs?",
        ]
        encoded = tokenizer(texts, padding=True, truncation=True)
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertEqual(len(encoded["input_ids"]), len(texts))

    @require_package("transformers")
    def test_tokenizer_with_model(self):
        """测试 tokenizer 编码后作为模型输入"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoTokenizer

        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        # 使用与 LLaMA tokenizer 兼容的 vocab_size (32000)
        config_dict = dict(SMALL_CONFIG)
        config_dict["vocab_size"] = 32000
        config = OpenELMConfig(**config_dict)
        model = OpenELMForCausalLM(config)
        model.eval()

        text = "Hello"
        encoded = tokenizer(text, return_tensors=None)
        input_ids = encoded["input_ids"]

        # 验证 token IDs 在模型 vocab 范围内
        for tid in input_ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, config.vocab_size)

        # 使用 tokenizer 输出作为模型输入
        input_tensor = paddle.to_tensor([input_ids], dtype="int64")
        with paddle.no_grad():
            output = model(input_ids=input_tensor, return_dict=True)

        self.assertEqual(
            list(output.logits.shape),
            [1, len(input_ids), config.vocab_size],
        )


class TestOpenELMInference(unittest.TestCase):
    """使用转换后的权重进行推理测试"""

    @slow
    def test_inference_with_converted_weights(self):
        """测试使用转换后的 Paddle 权重进行推理"""
        weight_path = os.path.join(_CONVERTED_WEIGHT_DIR, "model.pdparams")
        if not os.path.exists(weight_path):
            self.skipTest(f"转换后的权重文件不存在: {weight_path}")

        paddle.seed(_SEED)
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        # 使用 OpenELM-1_1B-Instruct 的配置创建模型
        config = OpenELMConfig(
            **{
                "vocab_size": 32000,
                "max_context_length": 2048,
                "num_transformer_layers": 28,
                "model_dim": 2048,
                "head_dim": 64,
                "qkv_multipliers": (0.5, 1.0),
                "num_gqa_groups": 4,
                "ffn_multipliers": (0.5, 4.0),
                "ffn_with_glu": True,
                "ffn_dim_divisor": 256,
                "activation_fn_name": "swish",
                "normalize_qk_projections": True,
                "share_input_output_layers": True,
                "rope_freq_constant": 10000,
                "rope_max_length": 4096,
                "initializer_range": 0.02,
                "use_cache": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
        model = OpenELMForCausalLM(config)

        # 手动加载权重
        state_dict = paddle.load(weight_path)
        model.set_state_dict(state_dict)
        model.eval()

        # 创建简单的输入
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        with paddle.no_grad():
            output = model(input_ids=input_ids, use_cache=False, return_dict=True)

        self.assertTrue(hasattr(output, "logits"))
        self.assertEqual(list(output.logits.shape), [1, 5, config.vocab_size])

        # 自回归生成
        cur_ids = input_ids.numpy()
        gen_ids = []
        max_gen_tokens = 10
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)

        self.assertGreater(len(gen_ids), 0, "模型没有生成任何 token")

    @slow
    @require_package("transformers")
    def test_inference_with_tokenizer(self):
        """测试使用 tokenizer 和转换后权重的完整推理流程"""
        weight_path = os.path.join(_CONVERTED_WEIGHT_DIR, "model.pdparams")
        if not os.path.exists(weight_path):
            self.skipTest(f"转换后的权重文件不存在: {weight_path}")

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from transformers import AutoTokenizer

        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        paddle.seed(_SEED)
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        # 使用 OpenELM-1_1B-Instruct 的配置创建模型并加载权重
        config = OpenELMConfig(
            **{
                "vocab_size": 32000,
                "max_context_length": 2048,
                "num_transformer_layers": 28,
                "model_dim": 2048,
                "head_dim": 64,
                "qkv_multipliers": (0.5, 1.0),
                "num_gqa_groups": 4,
                "ffn_multipliers": (0.5, 4.0),
                "ffn_with_glu": True,
                "ffn_dim_divisor": 256,
                "activation_fn_name": "swish",
                "normalize_qk_projections": True,
                "share_input_output_layers": True,
                "rope_freq_constant": 10000,
                "rope_max_length": 4096,
                "initializer_range": 0.02,
                "use_cache": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
        model = OpenELMForCausalLM(config)
        state_dict = paddle.load(weight_path)
        model.set_state_dict(state_dict)
        model.eval()

        encoded = tokenizer(_PROMPT_INFERENCE, return_tensors=None)
        input_ids_list = encoded["input_ids"]
        print(f"[Inference] prompt: {repr(_PROMPT_INFERENCE)}")

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        gen_ids = []
        max_gen_tokens = 32
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)
                if next_token == tokenizer.eos_token_id:
                    break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print("=" * 60)
        print(f"[Inference] 生成文本: {gen_text}")
        print("=" * 60)

        self.assertGreater(len(gen_ids), 0, "模型没有生成任何 token")
        self.assertGreater(len(gen_text.strip()), 0, "模型生成的文本为空")


if __name__ == "__main__":
    unittest.main()
