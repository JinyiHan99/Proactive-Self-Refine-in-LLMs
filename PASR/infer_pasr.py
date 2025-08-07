import json
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import httpx
import os
from openai import OpenAI, APITimeoutError
from config import prompt_config
os.environ["VLLM_USE_V1"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# with open("./system_prompt_refine.txt", "r", encoding="utf-8") as f:
#     system_prompt = f.read()

sampling_params_stop = SamplingParams(n=1, temperature=0.5, max_tokens=2048)

def gen_answers(vllm_gen, prompts, tokenizer):
    tip_text = []
    for x in prompts:
        # tip_text.append(first_prompt+x)
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": prompt_config["refine_prompt"]},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    voutputs = vllm_gen.generate(tip_text, sampling_params=sampling_params_stop, use_tqdm=False)
    first_answers = []
    for v in voutputs:
        for z in v.outputs:
            first_answers.append(z.text)

    return first_answers




def get_refine_answers(vllm_gen, dataset, output_path, problem_key, std_key, tokenizer):
    total_test = []
    
    for i in tqdm(range(0, len(dataset), 8)):
        items = dataset[i:i+8]
        problems = []
        ground_truths = []
        for item in items:
            problem, ground_truth = item[problem_key], item[std_key]
            ground_truths.append(ground_truth)
            problems.append(problem)
        first_answers = gen_answers(vllm_gen, problems, tokenizer)
        
        for problem, first_answer, ground_truth in zip(problems, first_answers, ground_truths):
            total_test.append({"question": problem, "std": ground_truth, "first_answer": first_answer})
        save_list_to_json(total_test, output_path)
    
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

# 主函数
def main(args):
    # i = 3500
        # 初始化模型和分词器
    for model_name in args.model_name:
        print(f"!!!!start to generate for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vllm_gen = LLM(model_name, gpu_memory_utilization=0.9)
        # save_name = f"ckp{i}"
        # i += 300
        try:
            for data_name in args.data_names:
                print(f"!!!!start to generate for {data_name}")
                # 这个数据集的 问题、标答 的key值，初始为空
                problem_key = ""
                std_key = ""
                # first_prompt = ""
                data_path = ""
                # 加载数据集
                dataset = []           
                data_path_file = args.data_path_file
                if data_name == "mmlu":
                    data_path = f"{data_path_file}/mmlu/select_mmlu_test.json"
                    problem_key = "Question"
                    std_key = "Ground_truth"
                    # first_prompt = gen_prompt_config['first_gen_mmlu']

                elif data_name == "humaneval":
                    data_path = f"{data_path_file}/humaneval/HumanEval.json"
                    problem_key = "prompt"
                    std_key = "canonical_solution"
                    # first_prompt = gen_prompt_config['first_gen_humaneval']

                elif data_name == "drop":
                    data_path = f"{data_path_file}/drop/test_sampled_313.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_drop']

                elif data_name == "xsum":
                    data_path = f"{data_path_file}/xsum/test_select313.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_xsum']

                elif data_name == "gsm8k":
                    data_path = f"{data_path_file}/gsm8k/test.jsonl"  # 替换为实际路径
                    problem_key="question"
                    std_key="answer"
                    # first_prompt = gen_prompt_config['first_gen_math']

                elif data_name == "math":
                    data_path = f"{data_path_file}/math/math.json"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_math']
            
                elif data_name == "aime24":
                    data_path = f"{data_path_file}/aime24/aime24.jsonl" 
                    # 根据数据集赋予相应的值
                    problem_key = "problem"
                    std_key = "solution"
                    # first_prompt = gen_prompt_config['first_gen_math']

                elif data_name == "arc-challenge":
                    data_path = f"{data_path_file}/arc/ARC-c/two_keys_test.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_arc']

                elif data_name == "gpqa":
                    data_path = f"{data_path_file}/gpqa/prompt_messages.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_gpqa']

                elif data_name == "wino":
                    data_path = f"{data_path_file}/winogrande/dev_two_keys.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_wino']

                elif data_name == "commonsenseqa":
                    data_path = f"{data_path_file}/commonsenseqa/two_keys_version.jsonl"
                    problem_key = "question"
                    std_key = "answer"
                    # first_prompt = gen_prompt_config['first_gen_commonsenseqa']

                dataset = load_dataset(data_path)
                output_path = f"{args.output_path}/{data_name}.json"
                directory = os.path.dirname(output_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    print(f"已创建目录: {directory}")
                print("output:", output_path)
                
                get_refine_answers(vllm_gen, dataset, output_path, problem_key, std_key, tokenizer)
        finally:
            del vllm_gen
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs='+', default=["/data2/models/THUDM/GLM-4-9B-0414"],
                        help="The name of the model")
    parser.add_argument("--output_path", type=str, default="/data2/WangXinyi/refine/baseline/self-refine/res/GLM-4-9B-0414/base",
                        help="The path of the output")
    parser.add_argument("--data_names", nargs='+', default=['math', 'gsm8k', 'aime24', 'mmlu', 'drop', 'xsum', 'arc-challenge', 'gpqa', 'wino', 'commonsenseqa'],
                        help="you can set multi data")
    parser.add_argument("--data_path_file", type=str, help= "your test data file path")


    args = parser.parse_args()
    main(args)