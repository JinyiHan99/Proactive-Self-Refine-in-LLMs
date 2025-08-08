import json
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import httpx
import os
from openai import OpenAI, APITimeoutError
from config import eval_prompt_config, revise_prompt_config, gen_prompt_config

import pdb

def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

# with open("./system_prompt_refine.txt", "r", encoding="utf-8") as f:
#     system_prompt = f.read()

sampling_params_stop = SamplingParams(n=1, temperature=0.6, max_tokens=1500)


def gen_answers(vllm_gen, tokenizer, prompts, first_answers, first_prompt):
    second_prompt = revise_prompt_config['second_revise_without_oracle']
    tip_text = []
    for x, first_ans in zip(prompts, first_answers):
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": first_prompt},
            {"role": "user", "content": x},
            {"role": "assistant", "content": first_ans},
            {"role": "user", "content": second_prompt}], tokenize=False, add_generation_prompt=True))

    # tip_text = ["what's your name?","what's your name?<|endoftext|>"]
    voutputs = vllm_gen.generate(tip_text, sampling_params=sampling_params_stop, use_tqdm=False)
    second_answers= []
    for v in voutputs:
        for z in v.outputs:
            second_answers.append(z.text)
    # pdb.set_trace()
    return first_answers,second_answers

def get_refine_answers(vllm_gen, tokenizer, dataset, output_path, first_prompt):
    total_test = []
    
    for i in tqdm(range(0, len(dataset), 8)):
        items = dataset[i:i+8]
        problems = []
        first_answers = []
        ground_truths = []
        for item in items:
            problem, ground_truth, first_answer= item["question"], item["std"], item['first_answer']
            ground_truths.append(ground_truth)
            problems.append(problem)
            first_answers.append(first_answer)
            first_answers, second_answers = gen_answers(vllm_gen, tokenizer, problems, first_answers, first_prompt)
        
        for problem, first_answer, second_answer, ground_truth in zip(problems, first_answers, second_answers, ground_truths):
            total_test.append({"question": problem, "std": ground_truth, "first_answer": first_answer, "second_answers_without_oracle":second_answer})
        save_list_to_json(total_test, output_path)
    
   

# 主函数
def main(args):
        # 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vllm_gen = LLM(args.model_name, gpu_memory_utilization=0.9, enable_prefix_caching = True)
    data_name_to_first_prompt = {
    "mmlu": gen_prompt_config['first_gen_mmlu'],
    "humaneval": gen_prompt_config['first_gen_humaneval'],
    "drop": gen_prompt_config['first_gen_drop'],
    "xsum": gen_prompt_config['first_gen_xsum'],
    "gsm8k": gen_prompt_config['first_gen_math'],
    "math": gen_prompt_config['first_gen_math'],
    "aime24": gen_prompt_config['first_gen_math'],
    "arc-challenge": gen_prompt_config['first_gen_arc'],
    "gpqa": gen_prompt_config['first_gen_gpqa'],
    "wino": gen_prompt_config['first_gen_wino'],
    "commonsenseqa": gen_prompt_config['first_gen_commonsenseqa']}
    for data_name in args.data_names:
        first_prompt = data_name_to_first_prompt[data_name]
         # 加载数据集
        data_path = f"{args.main_data_path}/{data_name}.json"
        if not data_path.endswith('_scores.json') and data_path.endswith('.json') and not data_path.endswith('oracle.json'):
            output_path = data_path.replace(".json", "_without_oracle.json")
        else:
            continue 
            # 跳过_scores.json文件
        ## Note: 在这里添加其他数据集
        data = read_json(data_path)

        # first_score_0_data = [item for item in data if item.get("first_score") == 0]

        # output_path = data_path.replace("_scores.json","_without_oracle.json")

             # 或者其他处理方式
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"已创建目录: {directory}")
        print("output:", output_path)
        ##### hjy：在这里你可以使用elif继续补充添加其他的data_path
        
        get_refine_answers(vllm_gen, tokenizer, data, output_path, first_prompt)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data2/models/THUDM/GLM-4-9B-0414",
                        help="The name of the model")
    parser.add_argument("--data_names", nargs='+', default=['math', 'gsm8k', 'aime24', 'mmlu', 'drop', 'xsum', 'arc-challenge', 'gpqa', 'wino', 'commonsenseqa'],
                        help="you can set multi data")
    parser.add_argument("--main_data_path", type=str, default = "/data2/WangXinyi/refine/baseline/self-refine/res/Qwen3-8B/with_without_oracle",
                        help="The main path to the data directory")
    args = parser.parse_args()
    main(args)