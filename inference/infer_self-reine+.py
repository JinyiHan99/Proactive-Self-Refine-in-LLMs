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

def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset



sampling_params_stop = SamplingParams(n=1, temperature=0.6, max_tokens=1500)


def gen_answers(vllm_gen, tokenizer, prompts, first_answers, first_prompt):
    second_prompt = revise_prompt_config['second_revise_w_oracle_general']
    tip_text = []
    for x, first_ans in zip(prompts, first_answers):
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": first_prompt},
            {"role": "user", "content": x},
            {"role": "assistant", "content": first_ans},
            {"role": "user", "content": second_prompt}], tokenize=False, add_generation_prompt=True))
    voutputs = vllm_gen.generate(tip_text, sampling_params=sampling_params_stop, use_tqdm=False)
    second_answers= []
    for v in voutputs:
        for z in v.outputs:
            second_answers.append(z.text)
    return first_answers,second_answers

def get_refine_answers(vllm_gen, tokenizer, dataset, output_path, first_prompt):
    total_test = []
    
    for i in tqdm(range(0, len(dataset), 8)):
        items = dataset[i:i+8]
        problems = []
        first_answers = []
        ground_truths = []
        for item in items:
            problem, ground_truth, first_answer= item["question"], item["std"], item['answer']
            ground_truths.append(ground_truth)
            problems.append(problem)
            first_answers.append(first_answer)
            first_answers, second_answers = gen_answers(vllm_gen, tokenizer,problems, first_answers, first_prompt)
        
        for problem, first_answer, second_answer, ground_truth in zip(problems, first_answers, second_answers, ground_truths):
            total_test.append({"question": problem, "std": ground_truth, "first_answer": first_answer, "second_answers_with_oracle":second_answer})
        save_list_to_json(total_test, output_path)
    
   


def main(args):

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

        data_path = f"{args.main_data_path}/{data_name}_scores.json"

        data = read_json(data_path)

        first_score_0_data = [item for item in data if item.get("score") == 0]

        output_path = data_path.replace("_scores.json","_with_oracle.json")
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        print("output:", output_path)

        
        get_refine_answers(vllm_gen, tokenizer, first_score_0_data, output_path, first_prompt)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="",
                        help="The name of the model")
    parser.add_argument("--data_names", nargs='+', default=['math', 'gsm8k', 'aime24', 'mmlu', 'drop', 'xsum', 'arc-challenge', 'gpqa', 'wino', 'commonsenseqa'],
                        help="you can set multi data")
    parser.add_argument("--main_data_path", type=str, default = "",
                        help="The main path to the data directory")
    args = parser.parse_args()
    main(args)