import pandas as pd

# 定义输入的 Parquet 文件路径和输出的 JSON 文件路径
parquet_file = './test-00000-of-00001.parquet'  # 替换为你的 Parquet 文件路径
json_file = 'aime24.json'           # 替换为你希望保存的 JSON 文件路径

# 读取 Parquet 文件
df = pd.read_parquet(parquet_file)

# 将 DataFrame 转换为 JSON 格式并保存
df.to_json(json_file, orient='records', lines=True)

print(f"Parquet 文件已成功转换为 JSON 文件，保存路径为: {json_file}")