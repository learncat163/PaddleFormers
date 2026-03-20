#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import paddle
from paddleformers.transformers.intern_lm2_5 import InternLM25Config, InternLM25ForCausalLM

def load_model_weights(model_path):
    ckpt_file = os.path.join(model_path, "model_state.pdparams")
    if os.path.exists(ckpt_file):
        return paddle.load(ckpt_file)
    
    chunk_files = sorted([f for f in os.listdir(model_path) if f.startswith("model_state.pdparams.")])
    if chunk_files:
        state_dict = {}
        for chunk_file in chunk_files:
            chunk_path = os.path.join(model_path, chunk_file)
            chunk_state = paddle.load(chunk_path)
            state_dict.update(chunk_state)
            print(f"Loaded chunk: {chunk_file} ({len(chunk_state)} tensors)")
        return state_dict
    
    raise FileNotFoundError(f"No model_state.pdparams or model_state.pdparams.* found in {model_path}")

config = InternLM25Config.from_pretrained('./tmp/internlm25_paddle_model_auto_shard')
model = InternLM25ForCausalLM(config)

state_dict = load_model_weights('./tmp/internlm25_paddle_model_auto_shard')
model.set_state_dict(state_dict)

print('Model loaded successfully!')
print(f'Total parameters: {sum(p.numel().item() for p in model.parameters()) / 1e9:.2f}B')