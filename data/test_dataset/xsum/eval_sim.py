# import json

# def calculate_avg_score(json_data):
#     """
#     计算 JSON 数据中 grade 的平均分，剔除 grade == -100 的元素。
    
#     :param json_data: 包含数据的 JSON 对象或列表
#     :return: 平均分 avg_score
#     """
#     # 确保输入是列表形式
#     if not isinstance(json_data, list):
#         raise ValueError("输入数据必须是列表形式")
    
#     # 剔除 grade == -100 的元素
#     filtered_data = [item for item in json_data if item.get("grade", -100) != -100]
    
#     # 如果过滤后没有数据，返回 None 或 0（根据需求）
#     if not filtered_data:
#         return 0.0
    
#     # 计算总分和数量
#     total_grade = sum(item.get("grade", 0) for item in filtered_data)
#     count = len(filtered_data)
    
#     # 计算平均分
#     avg_score = total_grade / count
#     return avg_score

# with open("/data2/WangXinyi/refine/eval/outcome/0330_v1/step_1800-xsum.json", "r", encoding="utf-8") as file:
#     json_data = json.load(file)
# avg_score = calculate_avg_score(json_data)
# print(f"平均分: {avg_score}")


import json

def read_json(file_path):
    """读取 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_average_scores(data):
    """计算 first_score 和 second_score 的平均值"""
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON 数据必须是一个非空列表")

    total_first_score = 0
    total_second_score = 0
    count = len(data)

    for item in data:
        # 确保每个字典都有 first_score 和 second_score 键
        if "first_score" not in item or "second_score" not in item:
            raise KeyError("每个数据项必须包含 'first_score' 和 'second_score' 键")
        
        total_first_score += item["first_score"]
        total_second_score += item["second_score"]

    # 计算平均值
    avg_first_score = total_first_score / count
    avg_second_score = total_second_score / count

    return avg_first_score, avg_second_score

# 主逻辑
json_file_path = "/data2/WangXinyi/refine/baseline/self-refine/res/res_0414/xsum_scores.json"  # 替换为你的 JSON 文件路径

# 读取 JSON 数据
data = read_json(json_file_path)

# 计算平均值
avg_first_score, avg_second_score = calculate_average_scores(data)

# 打印结果
print(f"Average of first_score: {avg_first_score}")
print(f"Average of second_score: {avg_second_score}")