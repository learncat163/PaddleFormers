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
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import paddle
from safetensors import deserialize as _deserialize, safe_open
from paddleformers.utils.serialization import _TYPES
from paddleformers.transformers.model_utils import shard_checkpoint
from paddleformers.utils.env import PADDLE_WEIGHTS_NAME, PADDLE_WEIGHTS_INDEX_NAME

from paddleformers.transformers.intern_lm2_5 import InternLM25Config, InternLM25Tokenizer

TRANSPOSE_WEIGHT_KEYS = ["wqkv", "wo", "w1", "w2", "w3", "output"]


def _needs_transpose(key):
    import re
    for trans_key in TRANSPOSE_WEIGHT_KEYS:
        if re.search(rf"\.{trans_key}\.weight$", key) or re.fullmatch(rf"^{trans_key}\.weight$", key):
            return True
    return False


def _load_safetensors_shard(shard_path, keep_bf16=False):
    with open(shard_path, "rb") as f:
        data = f.read()

    flat = _deserialize(data)
    weights = {}
    for key, meta in flat:
        if meta["dtype"] == "BF16":
            raw_uint16 = np.frombuffer(meta["data"], dtype=np.uint16).reshape(meta["shape"])
            if keep_bf16:
                weights[key] = raw_uint16.copy()
            else:
                arr_uint32 = raw_uint16.astype(np.uint32) << 16
                weights[key] = arr_uint32.view(np.float32).copy()
        else:
            dtype = _TYPES[meta["dtype"]]
            arr = np.frombuffer(meta["data"], dtype=dtype).reshape(meta["shape"])
            weights[key] = arr.astype(np.float32)

    return weights


def convert_hf_to_paddle(model_path, output_path, save_format="pd", num_layers=None, device="cpu", dtype="bfloat16"):
    model_path = os.path.expanduser(model_path)
    output_path = os.path.expanduser(output_path)

    paddle.set_device(device)

    print(f"Converting InternLM2.5 model from: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    if num_layers is not None:
        print(f"Num layers: {num_layers}")
    print("-" * 60)

    os.makedirs(output_path, exist_ok=True)

    print("Converting tokenizer...")
    tokenizer = InternLM25Tokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Tokenizer saved to {output_path}")

    print("Loading config...")
    config = InternLM25Config.from_pretrained(model_path)
    original_num_layers = config.num_hidden_layers
    if num_layers is not None:
        print(f"  Modifying num_hidden_layers: {config.num_hidden_layers} -> {num_layers}")
        config.num_hidden_layers = num_layers

    # 修正 model_type 为 internlm2_5
    # 原始 HuggingFace 模型使用 internlm2，但 PaddleFormers 使用 internlm2_5
    # 原始代码来源: configuration_utils.py to_dict() 方法会设置 model_type = self.__class__.model_type
    # 但如果实例的 __dict__ 中已有 model_type，to_dict() 会先复制 __dict__，然后用类属性覆盖
    # 这里需要手动从实例属性中移除旧的 model_type，确保保存时使用类属性
    if hasattr(config, "__dict__") and "model_type" in config.__dict__:
        del config.__dict__["model_type"]

    # 修正 auto_map 中的类名从 InternLM2 改为 InternLM25
    # 从 modeling_internlm2.InternLM2... 改为 modeling.InternLM25...
    if hasattr(config, "auto_map") and config.auto_map:
        corrected_auto_map = {}
        for key, value in config.auto_map.items():
            if "modeling_internlm2" in value:
                value = value.replace("modeling_internlm2", "modeling")
                value = value.replace("InternLM2", "InternLM25")
            # 确保 AutoConfig 指向 InternLM25Config
            if key == "AutoConfig" and "InternLM25Config" not in value:
                value = "configuration.InternLM25Config"
            corrected_auto_map[key] = value
        config.auto_map = corrected_auto_map

    # 确保 architectures 字段使用正确的类名
    if hasattr(config, "architectures") and config.architectures:
        config.architectures = [
            arch.replace("InternLM2", "InternLM25") for arch in config.architectures
        ]

    config.save_pretrained(output_path)
    print(f"  Config saved to {output_path}")

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        shard_files = [os.path.join(model_path, f) for f in os.listdir(model_path)
                       if f.endswith(".safetensors")]
        weight_map = {}
        for shard in shard_files:
            from safetensors import safe_open
            with safe_open(shard, framework="numpy") as f:
                for k in f.keys():
                    weight_map[k] = os.path.basename(shard)
    else:
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]

    target_layers = set(range(num_layers if num_layers is not None else original_num_layers))

    def _key_is_needed(key):
        if not key.startswith("model.layers."):
            return True
        parts = key.split(".")
        if len(parts) < 3:
            return True
        try:
            layer_idx = int(parts[2])
        except ValueError:
            return True
        return layer_idx in target_layers

    needed_keys = {k: v for k, v in weight_map.items() if _key_is_needed(k)}
    needed_shards = set(needed_keys.values())

    print(f"\nLoading weights from {len(needed_shards)} shard(s)...")
    print(f"  Total weight keys needed: {len(needed_keys)}")

    keys_by_shard = defaultdict(list)
    for key, shard_name in needed_keys.items():
        keys_by_shard[shard_name].append(key)

    keep_bf16 = (dtype == "bfloat16")
    paddle_state_dict = {}
    for shard_name in sorted(keys_by_shard.keys()):
        shard_path = os.path.join(model_path, shard_name)
        keys_in_shard = keys_by_shard[shard_name]
        print(f"  Loading shard: {shard_name} ({len(keys_in_shard)} keys)")
        shard_weights = _load_safetensors_shard(shard_path, keep_bf16=keep_bf16)
        for key in keys_in_shard:
            arr = shard_weights[key]
            if _needs_transpose(key) and arr.ndim == 2:
                arr = np.ascontiguousarray(arr.T)
            if keep_bf16 and arr.dtype == np.uint16:
                tensor = paddle.Tensor(arr, dtype="bfloat16", zero_copy=True)
            else:
                tensor = paddle.to_tensor(arr)
            paddle_state_dict[key] = tensor
        del shard_weights

    print(f"  Loaded {len(paddle_state_dict)} weight tensors")

    print(f"\nSaving PaddlePaddle model to {output_path}...")

    # 使用 PaddleFormers 标准分片函数，生成正确命名规范和 index 文件
    # 正确格式: model_state.pdparams (单片) 或
    #           model_state-00001-of-00003.pdparams + model_state.pdparams.index.json (多片)
    # 原始代码使用 model_state.pdparams.{idx} 格式，from_pretrained 无法识别
    shards, index = shard_checkpoint(paddle_state_dict, max_shard_size="2GB")
    for shard_file, shard in shards.items():
        shard_path = os.path.join(output_path, shard_file)
        paddle.save(shard, shard_path)
        total_bytes = sum(t.element_size() * t.numel().item() for t in shard.values())
        print(f"  Saved: {shard_file} ({len(shard)} tensors, {total_bytes / (1024 * 1024):.2f} MB)")

    if index is not None:
        index_path = os.path.join(output_path, PADDLE_WEIGHTS_INDEX_NAME)
        with open(index_path, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)
        print(f"  Saved index: {PADDLE_WEIGHTS_INDEX_NAME}")

    print(f"  Total {len(shards)} shard(s) saved")

    hf_gen_cfg = os.path.join(model_path, "generation_config.json")
    if os.path.exists(hf_gen_cfg):
        shutil.copy(hf_gen_cfg, os.path.join(output_path, "generation_config.json"))
        print("  generation_config.json copied from source")

    print("\nSaved files:")
    for fname in sorted(os.listdir(output_path)):
        fpath = os.path.join(output_path, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  - {fname}: {size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"PaddlePaddle model saved to: {output_path}")
    print(f"Dtype: {dtype}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert InternLM2.5 model from HuggingFace to PaddlePaddle format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/llm/internlm/internlm2_5-7b-chat/",
        help="Path to the HuggingFace model directory",
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
        help="Format to save the model",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of decoder layers to convert (default: all layers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Paddle device to use: 'cpu' or 'gpu' (default: 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Target weight dtype (default: bfloat16)",
    )

    args = parser.parse_args()

    hf_path = os.path.expanduser(args.model_path)
    if not os.path.exists(hf_path):
        print(f"ERROR: Model path does not exist: {hf_path}")
        return 1

    success = convert_hf_to_paddle(hf_path, args.output_path, args.save_format, args.num_layers, args.device, args.dtype)
    if not success:
        print("\nConversion failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
