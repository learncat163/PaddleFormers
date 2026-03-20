#!/usr/bin/env python
import os
import paddle

model_path = './tmp/internlm25_paddle_model_auto_shard'

chunk_files = sorted([f for f in os.listdir(model_path) if f.startswith("model_state.pdparams.")])
print(f"Found {len(chunk_files)} chunk files:")
for chunk_file in chunk_files:
    chunk_path = os.path.join(model_path, chunk_file)
    chunk_size = os.path.getsize(chunk_path) / (1024 * 1024)
    print(f"  {chunk_file}: {chunk_size:.2f} MB")

state_dict = {}
for chunk_file in chunk_files:
    chunk_path = os.path.join(model_path, chunk_file)
    chunk_state = paddle.load(chunk_path)
    state_dict.update(chunk_state)
    print(f"Loaded chunk: {chunk_file} ({len(chunk_state)} tensors)")

print(f"\nTotal tensors loaded: {len(state_dict)}")
print("Chunk loading test completed successfully!")