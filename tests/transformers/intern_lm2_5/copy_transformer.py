import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/cao/llm/internlm/internlm2_5-7b-chat/"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，导致显存不足
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 你好！有什么我可以帮助你的吗？
response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)