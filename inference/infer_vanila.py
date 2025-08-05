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




sampling_params_stop = SamplingParams(n=1, temperature=0, max_tokens=1500)

def gen_answers(vllm_gen, prompts, first_prompt):
    tip_text = []
    for x in prompts:
        tip_text.append(first_prompt+x)
    voutputs = vllm_gen.generate(tip_text, sampling_params=sampling_params_stop, use_tqdm=False)
    first_answers = []
    for v in voutputs:
        for z in v.outputs:
            first_answers.append(z.text)
    return first_answers

def get_refine_answers(vllm_gen, dataset, output_path, problem_key, std_key, first_prompt):
    total_test = []
    
    for i in tqdm(range(0, len(dataset), 8)):
        items = dataset[i:i+8]
        problems = []
        ground_truths = []
        for item in items:
            problem, ground_truth = item[problem_key], item[std_key]
            ground_truths.append(ground_truth)
            problems.append(problem)
        first_answers = gen_answers(vllm_gen, problems, first_prompt)
        
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


def main(args):

    vllm_gen = LLM(args.model_name, gpu_memory_utilization=0.7)

    for data_name in args.data_names:
        print(f"!!!!start to generate for {data_name}")

        problem_key = ""
        std_key = ""
        first_prompt = ""
        data_path = ""

        dataset = []           
        data_path_fold = args.data_path_fold
        if data_name == "mmlu":
            data_path = f"{data_path_fold}/mmlu/select_mmlu_test.json"
            problem_key = "Question"
            std_key = "Ground_truth"
            first_prompt = gen_prompt_config['first_gen_mmlu']

        elif data_name == "humaneval":
            data_path = f"{data_path_fold}/humaneval/HumanEval.json"
            problem_key = "prompt"
            std_key = "canonical_solution"
            first_prompt = gen_prompt_config['first_gen_humaneval']

        elif data_name == "drop":
            data_path = f"{data_path_fold}/drop/test_sampled_313.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_drop']

        elif data_name == "xsum":
            data_path = f"{data_path_fold}/xsum/test_select313.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_xsum']

        elif data_name == "gsm8k":
            data_path = f"{data_path_fold}/gsm8k/test.jsonl"  
            problem_key="question"
            std_key="answer"
            first_prompt = gen_prompt_config['first_gen_math']

        elif data_name == "math":
            data_path = f"{data_path_fold}/math/math.json"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_math']
    
        elif data_name == "aime24":
            data_path = f"{data_path_fold}/aime24/aime24.jsonl" 
            problem_key = "problem"
            std_key = "solution"
            first_prompt = gen_prompt_config['first_gen_math']

        elif data_name == "arc-challenge":
            data_path = f"{data_path_fold}/arc/ARC-c/two_keys_test.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_arc']

        elif data_name == "gpqa":
            data_path = f"{data_path_fold}/gpqa/prompt_messages.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_gpqa']

        elif data_name == "wino":
            data_path = f"{data_path_fold}/winogrande/dev_two_keys.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_wino']

        elif data_name == "commonsenseqa":
            data_path = f"{data_path_fold}/commonsenseqa/two_keys_version.jsonl"
            problem_key = "question"
            std_key = "answer"
            first_prompt = gen_prompt_config['first_gen_commonsenseqa']

        dataset = load_dataset(data_path)
        output_path = f"{args.output_path}/{data_name}.json"
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
      
        print("output:", output_path)
        
        get_refine_answers(vllm_gen, dataset, output_path, problem_key, std_key, first_prompt)
      

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="",
                        help="The name of the model")
    parser.add_argument("--data_names", nargs='+', default=['math', 'gsm8k', 'aime24', 'mmlu', 'drop', 'xsum', 'arc-challenge', 'gpqa', 'wino', 'commonsenseqa'],
                        help="you can set multi data")
    parser.add_argument("--output_path", type=str, default=""),
    parser.add_argument("--data_path_fold",type =str, help="here is the test data fold")
    args = parser.parse_args()
    main(args)