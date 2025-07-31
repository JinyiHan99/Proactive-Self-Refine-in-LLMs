import gzip
import json

def parse_jsonl_gz_to_json(input_file, output_file):
    """
    将 .jsonl.gz 文件解析并保存为标准的 JSON 文件。

    :param input_file: 输入的 .jsonl.gz 文件路径
    :param output_file: 输出的 JSON 文件路径
    """
    # 用于存储所有解析后的 JSON 对象
    data_list = []

    # 打开并解压 .jsonl.gz 文件
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            # 将每一行解析为 JSON 对象
            data = json.loads(line)
            data_list.append(data)

    # 将所有 JSON 对象写入到输出文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

# 示例调用
input_file = 'HumanEval.jsonl.gz'  # 输入文件路径
output_file = 'HumanEval.json'     # 输出文件路径
parse_jsonl_gz_to_json(input_file, output_file)

print(f"文件已成功解析并保存为 {output_file}")