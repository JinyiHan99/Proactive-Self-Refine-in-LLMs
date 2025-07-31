import json

# 输入和输出文件路径
input_file = "./dev.jsonl"  # 替换为你的输入文件路径
output_file = "./dev_two_keys.jsonl"  # 替换为你的输出文件路径

# 打开输入文件并逐行处理
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析 JSON 行
        data = json.loads(line)
        
        # 提取字段
        sentence = data["sentence"]
        option1 = data["option1"]
        option2 = data["option2"]
        answer = data["answer"]
        
        # 构造 question 字段
        question = f"{sentence}\nOption1: {option1}\nOption2: {option2}"
        
        # 构造新的 JSON 对象
        new_data = {
            "question": question,
            "answer": answer
        }
        
        # 写入输出文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print("转换完成！")