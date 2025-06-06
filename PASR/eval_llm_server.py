# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7,1'

import json
import random
import requests
import traceback
import time
from transformers import AutoTokenizer, AutoModelForCausalLM	
from transformers import GenerationConfig
from vllm import LLM, SamplingParams
from config import eval_config

from flask import Flask, request, Response

class Qwen_vllm(object):
    """docstring for Qwen"""
    def __init__(self, model='qwen2.5_32b'):
        print('当前模型:', model)
        if 'qwen2.5_32b' in model:
            model_name = "/data2/Qwen/Qwen2.5-32B-Instruct"
        
        elif 'qwen2.5_7b' in model:
            model_name = "/data2/Qwen/Qwen2.5-7B-Instruct"
    
        elif 'qwen2.5_14b' in model:
            model_name = "/data2/Qwen/Qwen2.5-14B-Instruct"
          
        elif 'llama3.1_8b' in model:
            model_name = "/data2/models/Meta-Llama-3.1-8B-Instruct"
        else:
            print('未考虑到的模型', model)
            exit()
        print('model_name:', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.vllm = LLM(model = model_name, gpu_memory_utilization=0.8, tensor_parallel_size=4) 
    
    def get_gen_response(self, queries):
        prompts = []
        for x in queries:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": x}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
        all_res = []
        outputs = self.vllm.generate(prompts, SamplingParams(temperature=0))
        for output in outputs:
            generate_text = output.outputs[0].text
            all_res.append(generate_text)
        return all_res
    
    def get_eval_response(self, data):
        sys_prompt = '''
            You are a judger, you will judge the correctness of the answer to the question.
            Below is a question, a ground truth answer, and an answer generated by an AI assistant, 
            please rate the AI assistant's answers according to the question on a scale from 0 to 1.
            Your output is just a number in the range from 0 to 1.
            '''
        usr_prompt = '''
            ### Question:
            {Question}

            ### Ground Truth:
            {Ground_Truth}

            ### Answer:
            {Answer}
            '''
        prompts = []
        for x in data:
            print("!!eval_question:",x['q'])
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content":  usr_prompt.format(Question = x['q'], Ground_Truth = x['std'], Answer = x['answer'])}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
        all_res = []
        outputs = self.vllm.generate(prompts, SamplingParams(temperature=0))
        for output in outputs:
            generate_text = output.outputs[0].text
            all_res.append(generate_text)
        return all_res

llm = Qwen_vllm('qwen2.5_32b')
app = Flask(__name__)  # 初始化app
@app.route('/generate', methods=["POST"])
def handle():
    data = json.loads(request.get_data())
    result = json.dumps(llm.get_eval_response(data))
    response = Response(result, mimetype="application/json")
    return response

if __name__ == '__main__':
    app.run(host='localhost', port=eval_config['eval_llm_port'])
    
