import re
import json

input_file = "/data2/WangXinyi/refine/datasets/drop/test_select.jsonl"
output_file = "/data2/WangXinyi/refine/datasets/drop/test_select_fixed.jsonl"

with open(input_file, "r") as f:
    content = f.read()

# 正则匹配每个 JSON 对象（以 {"question": 开始，直到下一个 {"question": 或文件结束）
# 注意：这个写法假设没有嵌套 {}} 的情况（对你这类数据是安全的）
items = re.findall(r'\{(?:"question":.*?)(?=\n?\{"question":|\Z)', content, flags=re.DOTALL)

with open(output_file, "w") as f:
    for item in items:
        item = "{" + item.strip()
        try:
            json.loads(item)  # 验证是合法 JSON
            f.write(item + "\n")
        except json.JSONDecodeError as e:
            print("[跳过非法JSON]", e)
