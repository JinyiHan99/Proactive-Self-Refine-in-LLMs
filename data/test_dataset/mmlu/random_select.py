import json
import random

# 定义文件路径
input_file = './merged_mmlu_test.json'  # 输入的 JSON 文件路径
output_file = './select_mmlu_test.json'  # 输出的 JSON 文件路径

# 读取 JSON 文件
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {input_file} 不存在，请检查路径是否正确。")
except json.JSONDecodeError:
    raise ValueError(f"文件 {input_file} 格式错误，请确保是有效的 JSON 文件。")

# 检查数据是否为列表
if not isinstance(data, list):
    raise ValueError("JSON 文件的内容必须是一个数组 (list)。")

# 确保数据量足够
num_to_select = 100
if len(data) < num_to_select:
    raise ValueError(f"JSON 文件中的数据不足 {num_to_select} 条。")

# 随机选择 100 条数据
selected_data = random.sample(data, num_to_select)

# 将结果保存到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=4)

print(f"已成功从 {input_file} 中随机筛选 {num_to_select} 条数据，并保存到 {output_file}。")