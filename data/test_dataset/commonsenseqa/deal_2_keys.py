import json

# 输入和输出文件路径
input_file = "./dev_rand_split.jsonl"  # 替换为你的输入文件路径
output_file = "./two_keys_version.jsonl"  # 替换为你的输出文件路径

# 打开输入文件并逐行处理
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析 JSON 行
        data = json.loads(line)
        
        # 提取问题的 stem 和选项
        stem = data["question"]["stem"]
        choices = data["question"]["choices"]
        
        # 构造 question 字段
        question_parts = [stem]
        for choice in choices:
            question_parts.append(f"{choice['label']}: {choice['text']}")
        question = "\n".join(question_parts)
        
        # 提取 answer 字段
        answer = data["answerKey"]
        
        # 构造新的 JSON 对象
        new_data = {
            "question": question,
            "answer": answer
        }
        
        # 写入输出文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print("转换完成！")