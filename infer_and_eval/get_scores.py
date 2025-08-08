import os
import json

def calculate_avg_scores(folder_path):
    # 遍历文件夹中的所有文件
    avg_score_all = 0
    i = 0
    for filename in os.listdir(folder_path):
        # 检查文件是否以 "_scores.json" 结尾
        if filename.endswith("_scores.json"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 打开并加载 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # 提取所有 score 值
                scores = [item['score'] for item in data if 'score' in item]
                
                # 计算平均分
                if scores:
                    avg_score = sum(scores) / len(scores)
                    avg_score_all += avg_score
                    i += 1
                else:
                    avg_score = 0  # 如果没有 score 数据，默认平均分为 0
                
                # 打印文件名和平均分
                print(f"File: {filename}, Average Score: {avg_score:.3f}")
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    avg_score_all = avg_score_all / i
    print(f'总平均分:{avg_score_all:.3f}')

# 使用示例
folder_path = "/mnt/data/kw/wxy/self_refine/Qwen2.5-14B/PTR"  # 替换为你的文件夹路径
calculate_avg_scores(folder_path)