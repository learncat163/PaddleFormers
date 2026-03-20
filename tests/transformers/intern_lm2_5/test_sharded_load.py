#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import paddle
from paddleformers.transformers.model_utils import load_sharded_checkpoint
from paddleformers.transformers.intern_lm2_5 import InternLM25Config, InternLM25ForCausalLM

config = InternLM25Config.from_pretrained('./tmp/internlm25_paddle_model_auto_shard')
model = InternLM25ForCausalLM(config)
load_sharded_checkpoint(model, './tmp/internlm25_paddle_model_auto_shard', strict=False, prefer_safe=False)
print('Model loaded successfully!')
print(f'Total parameters: {sum(p.numel().item() for p in model.parameters()) / 1e9:.2f}B')