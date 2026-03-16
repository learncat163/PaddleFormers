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
Standalone worker: PaddlePaddle forward pass -> save logits to .npy file.

Designed to be run as an isolated subprocess so GPU VRAM is fully released when
this process exits.

Key design points:
  - paddle.set_device(device) is called FIRST so all parameters land on GPU directly.
  - paddle.LazyGuard() wraps model construction to suppress random weight init,
    avoiding the ~28 GB GPU allocation that would otherwise OOM before loading the
    real checkpoint (bfloat16 weights cast to float32 ~= 28 GB on GPU).
  - After LazyGuard, set_state_dict() materializes the real weights on GPU.

Original references:
  - paddle.LazyGuard: skips _initialize_weights / _init_weights during __init__
  - paddleformers/transformers/model_utils.py _post_init / init_weights

Usage:
    python diff_worker_paddle.py \\
        --model_path ./tmp/internlm25_paddle_model \\
        --save_path /tmp/logits_paddle.npy \\
        --device gpu
"""

import argparse
import gc
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Paddle forward worker for diff comparison")
    parser.add_argument("--model_path", required=True, help="Converted PaddlePaddle model directory")
    parser.add_argument("--save_path", required=True, help="Path to save logits (.npy)")
    parser.add_argument(
        "--device",
        default="gpu",
        help="Paddle device: 'gpu' or 'cpu'",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--input_ids",
        default="1,345,232,328,740,140,1695,69,6078,1588",
        help="Comma-separated input token IDs",
    )
    args = parser.parse_args()

    # Add project root to sys.path so paddleformers is importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    model_path = os.path.expanduser(args.model_path)
    save_path = os.path.expanduser(args.save_path)
    ckpt_path = os.path.join(model_path, "model_state.pdparams")
    input_ids = list(map(int, args.input_ids.split(",")))

    import paddle
    from paddleformers.transformers.intern_lm2_5 import (
        InternLM25Config,
        InternLM25ForCausalLM,
        InternLM25PretrainedModel,
    )

    # Set device FIRST so all subsequent paddle operations land on the target device
    paddle.set_device(args.device)
    paddle.seed(args.seed)

    fixed_input = np.array([input_ids], dtype=np.int64)

    config = InternLM25Config.from_pretrained(model_path)
    print(f"[paddle worker] num_hidden_layers={config.num_hidden_layers}", flush=True)
    print(f"[paddle worker] device={args.device}", flush=True)

    # Use bfloat16 throughout to avoid OOM:
    #   - float32 model on GPU = ~28GB, which together with the ~14GB bfloat16
    #     state_dict loaded from disk totals ~42GB (OOM on a 30.5GB card).
    #   - bfloat16 model = ~14GB; state_dict = ~14GB (peak during set_state_dict),
    #     which safely fits in 30.5GB VRAM.
    # Only the final logits are cast to float32 for numpy comparison.
    # Original dtype discussion: convert_weight_to_paddle.py _run_paddle_forward
    paddle.set_default_dtype("bfloat16")

    # LazyGuard creates lazy tensors (zero memory). InternLM25PretrainedModel._init_weights
    # calls module.weight[padding_idx].zero_() which fails on a zero-memory lazy tensor.
    # Monkey-patch _init_weights on the BASE CLASS to be a no-op so model construction
    # succeeds without allocating any GPU VRAM for random weights.
    # Weights are populated by set_state_dict() immediately after.
    # Original: paddleformers/transformers/intern_lm2_5/modeling.py InternLM25PretrainedModel._init_weights
    print("[paddle worker] creating model in bfloat16 (no random init)...", flush=True)
    _orig_init_weights = InternLM25PretrainedModel._init_weights
    InternLM25PretrainedModel._init_weights = lambda self, layer: None
    try:
        with paddle.LazyGuard():
            model = InternLM25ForCausalLM(config)
    finally:
        InternLM25PretrainedModel._init_weights = _orig_init_weights

    print("[paddle worker] loading checkpoint...", flush=True)
    # paddle.load loads tensors onto whatever place (defaults to CPU if no device set).
    # We do NOT cast bf16 -> f32 here to avoid peak VRAM spike of 14+28=42GB.
    # Both model params and state_dict stay in bfloat16.
    state_dict = paddle.load(ckpt_path)
    model.set_state_dict(state_dict)
    del state_dict
    gc.collect()

    model.eval()
    print("[paddle worker] model ready (bfloat16 on " + args.device + ").", flush=True)

    paddle_input = paddle.to_tensor(fixed_input, dtype=paddle.int64)
    with paddle.no_grad():
        out = model(input_ids=paddle_input)

    # out is a tuple; first element is logits [B, seq_len, vocab_size] before Softmax.
    # Cast bfloat16 -> float32 only for the final comparison output.
    logits_tensor = out[0] if isinstance(out, (tuple, list)) else out.logits
    logits = logits_tensor.cast("float32").cpu().numpy()

    print(f"[paddle worker] logits shape={logits.shape}  dtype=float32", flush=True)
    np.save(save_path, logits)
    print(f"[paddle worker] saved to {save_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
