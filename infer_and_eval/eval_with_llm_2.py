
from config import eval_prompt_config, revise_prompt_config, gen_prompt_config
import json
from tqdm import tqdm
import re
import requests
import pdb
from openai import OpenAI
def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

# query_server = "http://localhost:59821"
# def reward_correct(contents):
#     """
#     使用卡或者API来判断生成的答案是否正确。
#     """
#     scores = requests.post(url=f"{query_server}/generate", json = contents).json() 
#     final_scores = []
#     for res in scores:
#         try: 
#             reward = float(res)
#         except:
#             numbers = re.findall(r"[-+]?\d*\.\d+|\d+", res)
#             if numbers:
#                 # 取最后一个数字并转换
#                 last_number = numbers[-1]
#                 reward = float(last_number)
#             else:
#                 # 如果没有找到任何数字，返回0
#                 reward = 0.0
#         finally:
#             # 确保奖励在0-1之间
#             reward = max(0.0, min(reward, 1.0))
#         final_scores.append(reward)
#     return final_scores

client = OpenAI(base_url="http://0.0.0.0:59825/v1/", api_key="dummy")
prompt_template = '''
        ### Question:
        {Question}

        ### Ground Truth:
        {Ground_Truth}

        ### Answer:
        {Answer}
        '''
def llm_eval(sys_prompt, example):
    user_prompt = prompt_template.format(Question = example['question'], Ground_Truth = example['std'], Answer = example['first_answer'])
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}],
        temperature=0.2,
        )
    res = response.choices[0].message.content
    try: 
        reward = float(res)

    except:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", res)
        if numbers:
        # 取最后一个数字并转换
            last_number = numbers[-1]
            reward = float(last_number)
        else:
        # 如果没有找到任何数字，返回0
            reward = 0.0
    finally:
    # 确保奖励在0-1之间
        reward = max(0.0, min(reward, 1.0))
    return reward

def main(args):
    DATA_EVAL_MAPPING = {
    "gsm8k": "general_type",
    "aime24": "general_type",
    "math": "general_type",
    "mmlu": "multi_choice_type",
    "humaneval": "general_type",
    "drop": "general_type",
    "xsum": "sum_type",
    "arc-challenge": "multi_choice_type",
    "gpqa": "general_type",
    "wino": "general_type",
    "commonsenseqa": "general_type",
}


    for data_name in args.data_names:
        print(f"!!start to gen scores for {data_name}")
        eval_type = DATA_EVAL_MAPPING.get(data_name, None)
        sys_prompt = eval_prompt_config[eval_type]          
        data_path = f"{args.main_data_path}/{data_name}.json"
        print(data_path)
        dataset = read_json(data_path)
        output_path = data_path.replace(".json","_scores.json")
        total_test =[]
        for batch_start in tqdm(range(0,len(dataset),32)):
            temp_data = dataset[batch_start: batch_start+32]
            cur_eval_data = []
            for example in temp_data:
                score = llm_eval(sys_prompt, example)
                cur_eval_data.append({"question":example['question'], "std":example['std'], "answer":example['first_answer'], "score":score})
            #判断模型第一次答案的得分情况
            # scores = llm_eval(cur_eval_data)
            #记录每一条数据的得分
            # cur_eval_data_temp = []
            # for single_data, score in zip(temp_data, scores):
            #     single_data['score'] = score

            #     cur_eval_data_temp.append(single_data)

            total_test.extend(cur_eval_data)

            save_list_to_json(total_test, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", nargs='+', default=['math', 'gsm8k', 'aime24', 'mmlu', 'drop', 'arc-challenge', 'gpqa', 'wino', 'commonsenseqa','xsum'],
                        help="The data name you need")
    parser.add_argument("--main_data_path", type=str, default = "/data2/WangXinyi/refine/baseline/self-refine/res/GLM-4-9B-0414/base",
                        help="The main path to the data directory")
    args = parser.parse_args()
    main(args)