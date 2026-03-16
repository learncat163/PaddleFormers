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

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import paddle
from paddleformers.transformers.intern_lm2_5 import (
    InternLM25Config,
    InternLM25ForCausalLM,
    InternLM25Tokenizer,
)


def main():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    model_path = os.path.join(project_root, "tmp", "internlm25_paddle_model")

    paddle.set_device("gpu")
    paddle.set_default_dtype("bfloat16")

    config = InternLM25Config.from_pretrained(model_path)
    model = InternLM25ForCausalLM(config)
    model.set_state_dict(paddle.load(os.path.join(model_path, "model_state.pdparams")))
    model.eval()

    tokenizer = InternLM25Tokenizer.from_pretrained(model_path, load_checkpoint_format="")

    prompt = "猫和狗的区别是什么"
    chat_inputs = model.build_inputs(
        tokenizer, prompt, history=[], meta_instruction="You are a helpful assistant."
    )

    with paddle.no_grad():
        out = model.generate(
            input_ids=chat_inputs["input_ids"],
            attention_mask=chat_inputs.get("attention_mask"),
            max_new_tokens=512,
            use_cache=True,
            decode_strategy="greedy_search",
        )

    seq = out[0] if isinstance(out, (list, tuple)) else out
    print(tokenizer.decode(seq.numpy().tolist()[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
