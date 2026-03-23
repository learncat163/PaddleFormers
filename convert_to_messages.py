#!/usr/bin/env python3
"""
将 paddleformers 格式的数据转换为 messages 格式
"""

import json
import os

def convert_to_messages(input_file, output_file):
    """将 paddleformers 格式转换为 messages 格式"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line.strip())

            # 转换为 messages 格式
            messages = [
                {"role": "user", "content": data['src']},
                {"role": "assistant", "content": data['tgt']}
            ]

            # 写入新格式
            new_data = {"messages": messages}
            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

def main():
    # 转换训练集
    train_input = "./tmp/gsm8k/gsm8k_train.jsonl"
    train_output = "./tmp/gsm8k/gsm8k_train_messages.jsonl"

    if os.path.exists(train_input):
        print(f"转换训练集: {train_input} -> {train_output}")
        convert_to_messages(train_input, train_output)
        print("训练集转换完成")

    # 转换验证集
    eval_input = "./tmp/gsm8k/gsm8k_eval.jsonl"
    eval_output = "./tmp/gsm8k/gsm8k_eval_messages.jsonl"

    if os.path.exists(eval_input):
        print(f"转换验证集: {eval_input} -> {eval_output}")
        convert_to_messages(eval_input, eval_output)
        print("验证集转换完成")

if __name__ == "__main__":
    main()