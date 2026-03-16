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
Standalone worker: HuggingFace (PyTorch) forward pass -> save logits to .npy file.

Designed to be run as an isolated subprocess so GPU VRAM is fully released when
this process exits, before the PaddlePaddle worker starts.

Usage:
    python diff_worker_torch.py \\
        --model_path ~/llm/internlm/internlm2_5-7b-chat \\
        --save_path /tmp/logits_torch.npy \\
        --device gpu
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Torch forward worker for diff comparison")
    parser.add_argument("--model_path", required=True, help="HuggingFace model directory")
    parser.add_argument("--save_path", required=True, help="Path to save logits (.npy)")
    parser.add_argument(
        "--device",
        default="gpu",
        help="torch device: 'gpu' (maps to 'cuda') or 'cpu'",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--input_ids",
        default="1,345,232,328,740,140,1695,69,6078,1588",
        help="Comma-separated input token IDs",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    save_path = os.path.expanduser(args.save_path)
    torch_device = "cuda" if args.device in ("gpu", "cuda") else "cpu"
    input_ids = list(map(int, args.input_ids.split(",")))

    import torch

    torch.manual_seed(args.seed)
    if torch_device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    fixed_input = np.array([input_ids], dtype=np.int64)

    from transformers import AutoModelForCausalLM

    print(f"[torch worker] loading model from {model_path} on {torch_device}...", flush=True)
    # Use bfloat16 to match the paddle worker dtype so the diff reflects weight
    # conversion accuracy rather than float32-vs-bfloat16 precision differences.
    # Original: torch_dtype=torch.float32 (changed to bfloat16 for fair comparison)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()
    print(f"[torch worker] loaded. num_hidden_layers={model.config.num_hidden_layers}", flush=True)

    inp = torch.tensor(fixed_input, dtype=torch.long).to(torch_device)
    with torch.no_grad():
        out = model(input_ids=inp)

    # .logits is [B, seq_len, vocab_size], output of lm_head BEFORE any Softmax
    logits = out.logits.float().cpu().numpy()
    print(f"[torch worker] logits shape={logits.shape}  dtype=float32", flush=True)

    np.save(save_path, logits)
    print(f"[torch worker] saved to {save_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
