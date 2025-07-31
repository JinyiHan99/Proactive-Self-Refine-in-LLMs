import json

def jsonl_to_json(input_file, output_file):
    """
    将 JSONL 文件转换为标准 JSON 文件。
    
    :param input_file: 输入的 JSONL 文件路径
    :param output_file: 输出的 JSON 文件路径
    """
    try:
        # 读取 JSONL 文件并解析每一行
        with open(input_file, 'r', encoding='utf-8') as infile:
            json_objects = [json.loads(line) for line in infile]

        # 将所有 JSON 对象写入一个数组，并保存为标准 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(json_objects, outfile, ensure_ascii=False, indent=4)

        print(f"转换成功！已将 {input_file} 转换为 {output_file}")

    except Exception as e:
        print(f"发生错误：{e}")

# 示例用法
if __name__ == "__main__":
    input_path = "./aime24.jsonl"  # 输入的 JSONL 文件路径
    output_path = "./aime24.json"  # 输出的 JSON 文件路径
    jsonl_to_json(input_path, output_path)