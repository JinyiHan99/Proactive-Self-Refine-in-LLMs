import pandas as pd
import json
import os

# 读取 CSV 文件并转换为 JSON 格式
def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_csv_folder(csv_folder_path, output_json_path):
    total = []
    
    # 遍历文件夹中的每个 CSV 文件
    for csv_file in os.listdir(csv_folder_path):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(csv_folder_path, csv_file)
            df = pd.read_csv(csv_file_path)
            
            for i in range(len(df)):
                temp = {}
                # 确保所有数据都转换为字符串
                question = str(df.iloc[i][0]) + "\n" + "A. " + str(df.iloc[i][1]) + "\n" + "B. " + str(df.iloc[i][2]) + "\n" + "C. " + str(df.iloc[i][3]) + "\n" + "D. " + str(df.iloc[i][4])
                ground_truth = str(df.iloc[i][5])
                
                temp['Question'] = question
                temp['Ground_truth'] = ground_truth
                total.append(temp)
    
    # 转换为 JSON 格式并保存
    save_list_to_json(total, output_json_path)

# 输入文件夹路径和输出 JSON 文件路径
csv_folder_path = "/data2/WangXinyi/refine/datasets/mmlu/test"
output_json_path = "/data2/WangXinyi/refine/datasets/mmlu/test/merged_mmlu_test.json"

process_csv_folder(csv_folder_path, output_json_path)
