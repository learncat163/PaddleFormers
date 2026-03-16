#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原版 InternLM2.5 模型推理脚本
使用 CPU 模式，兼容 transformers 4.55.4
"""
import os

# 使用展开的路径，避免硬编码 /home/cao
model_path = os.path.expanduser("~/llm/internlm/internlm2_5-7b-chat/")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置 CPU 模式
torch.set_num_threads(1)

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Tokenizer 加载完成")

# 使用 CPU 模式加载模型，使用 float32 精度
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
).cpu()
model = model.eval()
print("模型加载完成")

# 简单推理测试
print("\n" + "=" * 60)
print("运行推理测试")
print("=" * 60)

# 准备固定输入进行测试
fixed_input_ids = [[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588]]
print(f"\n[输入] input_ids: {fixed_input_ids}")

# 转换为 tensor
input_tensor = torch.tensor(fixed_input_ids, dtype=torch.long)
attention_mask = torch.ones_like(input_tensor)

# 获取第一层输出进行测试
class HookCapture:
    def __init__(self):
        self.output = None

    def hook(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0]
        else:
            self.output = output

capture = HookCapture()
hook_handle = model.model.layers[0].register_forward_hook(capture.hook)

print("\n运行前向推理...")
with torch.no_grad():
    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask
    )

hook_handle.remove()

layer0_output = capture.output
print(f"\n[输出] 第一层输出 shape: {layer0_output.shape}")
print(f"[输出] 第一层输出统计:")
print(f"  - mean: {layer0_output.mean().item():.6f}")
print(f"  - std: {layer0_output.std().item():.6f}")
print(f"  - min: {layer0_output.min().item():.6f}")
print(f"  - max: {layer0_output.max().item():.6f}")

# 检查第一层权重
print(f"\n[权重] model.layers.0.attention.wqkv.weight:")
wqkv_weight = model.state_dict()["model.layers.0.attention.wqkv.weight"]
print(f"  - shape: {wqkv_weight.shape}")
print(f"  - dtype: {wqkv_weight.dtype}")
print(f"  - mean: {wqkv_weight.mean().item():.6f}")
print(f"  - std: {wqkv_weight.std().item():.6f}")
print(f"  - min: {wqkv_weight.min().item():.6f}")
print(f"  - max: {wqkv_weight.max().item():.6f}")

print("\n" + "=" * 60)
print("推理测试完成")
print("=" * 60)
