import json
import random
import re
input_file = "/data2/WangXinyi/refine/datasets/drop/test.jsonl"
output_file = "/data2/WangXinyi/refine/datasets/drop/test_sampled_313.jsonl"
sample_size = 313

# 读取整个文件内容（可能存在一行多个 JSON 对象）
with open(input_file, "r") as f:
    content = f.read()

# 使用正则提取所有 JSON 对象（每个以 {"question": 开头）
raw_jsons = re.findall(r'\{(?:"question":.*?)}', content, flags=re.DOTALL)

# 尝试解析为 dict，并去重
parsed = []
seen = set()
for item in raw_jsons:
    try:
        obj = json.loads(item)
        uid = json.dumps(obj, sort_keys=True)
        if uid not in seen:
            parsed.append(obj)
            seen.add(uid)
    except json.JSONDecodeError as e:
        print(f"[跳过非法 JSON] {e}")

# 抽样
sampled = random.sample(parsed, min(sample_size, len(parsed)))

# 写入结果
with open(output_file, "w") as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已抽取 {len(sampled)} 条样本，输出到: {output_file}")
