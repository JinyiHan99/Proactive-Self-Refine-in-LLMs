from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, random
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import wandb,traceback
import pdb
from config import train_config, ds_config, prompt_config, eval_config
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
from transformers import GenerationConfig
from openai import OpenAI, APITimeoutError
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["NCCL_P2P_DISABLE"]="1"
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ["VLLM_USE_V1"] = "0"

# wandb_name = train_config['wandb_name']
# wandb_project = train_config['wandb_project']
# wandb_key = train_config['wandb_key']
model_path = train_config['model_path']
save_path = train_config['save_path']
record_path = train_config['record_path']
gen_data_path = train_config['gen_data_path']
gen_device = train_config['gen_device']   
all_steps = train_config['all_steps']
Q_batch_size = train_config['Q_batch_size']
num_pre_Q = train_config['num_pre_Q']
train_batch_size = train_config['train_batch_size']
gen_update_steps = train_config['gen_update_steps']
save_steps = train_config['save_steps']
compute_gen_logps = train_config['compute_gen_logps']
clip_param = train_config['clip_param']
ref_server = train_config['ref_server']
beta = train_config['beta']
program_path = train_config['gen_data_path']
global update_model_num
update_model_num = 0
eval_query_server = f"http://localhost:{eval_config['eval_llm_port']}"
sys_prompt = train_config['eval_prompt']

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    data['gen_logps'] = bytes_to_tensor(dd[4])
    data['acc_scores'] = bytes_to_tensor(dd[5])
    data['format_scores'] = bytes_to_tensor(dd[6])

    # print("!!!batch的key", data.keys())
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    print(f"!!! rank: {torch.distributed.get_rank()} inputs shape: {inputs.shape} ")
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the logits successfully!!")

    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).float()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the loss successfully!!")
    return loss

import signal
import time

def handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.7)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    #sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=600)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
    data_path = train_config['data_path']
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    QAs = [{'Q': item['instruction'], 'A': item['output']} for item in dataset] 
   
    system_prompt_refine = prompt_config['refine_prompt']
    system_prompt_prev = prompt_config['raw_prompt']
    sampling_params = SamplingParams(n= num_pre_Q, temperature=0.9, max_tokens=1500)

    
    def gen_answers(prompts, system_prompt):
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        # answers = get_completions(tip_text,0)
        voutputs = vllm_gen.generate(tip_text, sampling_params = sampling_params, use_tqdm=False)
        answers = []
        for v in voutputs:
            for z in v.outputs:
                answers.append(z.text)

        return answers

    def reward_format(answer):
        pattern = r"^<think>.*?</think>[\n ]*?<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        # 先检查是否符合格式的要求
        if not re.match(pattern, answer, re.DOTALL | re.VERBOSE) or think_count != 2 and answer_count != 2:
            return -1
        #再去检查refine标签是否在think标签中，如果不在，则返回-1， 否则返回1
        think_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
        if not think_match:
            return -1
        think_start, think_end = think_match.span()
        # 第三步：查找所有 <refine> 标签并检查是否都在 <think> 内
        for match in re.finditer(r"<refine>(.*?)</refine>", answer, re.DOTALL):
            start, end = match.span()
            if not (think_start <= start and end <= think_end):
                return -1
        return 1

    def reward_refine(prev_answer, refined_answer, acc_scores, pre_scores):
        refined_scores = []
        pre_score_avg = sum(pre_scores) / len(pre_scores) if len(pre_scores)>0 else 0
        for i in range(len(prev_answer)):
            start_refine_cnt = refined_answer[i].count("<refine>")
            end_refine_cnt = refined_answer[i].count("</refine>")
            if start_refine_cnt == end_refine_cnt and start_refine_cnt > 0: 
                if acc_scores[i] > pre_score_avg:
                    refine_score = 1.0
                elif acc_scores[i] < pre_score_avg:
                    refine_score = -1.0
                else:
                    refine_score = -0.5
            else:
                    refine_score = 0.0
            refined_scores.append(refine_score)
        return refined_scores

    prompt_template = '''
            ### Question:
            {Question}

            ### Ground Truth:
            {Ground_Truth}

            ### Answer:
            {Answer}
            '''
    client = OpenAI(base_url="http://10.176.64.144:8858/v1/", api_key="dummy")

    def llm_eval(ans, example):
        user_prompt = prompt_template.format(Question = example['q'], Ground_Truth = example['std'], Answer = ans)
        response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=[
                {"role": "system", "content": '''You are a judger, you will judge the correctness of the answer to the question. Below is a question, a ground truth answer, and an answer generated by an AI assistant, please rate the AI assistant's answers according to the question on a scale from 0 to 1. Your output is just a number in the range from 0 to 1.'''},
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

    
    

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        refined_answers = gen_answers(prompts, system_prompt_refine)
        prev_answers = gen_answers(prompts, system_prompt_prev)
        rewards = []
        scores = []
        record_gen= []
        acc_scores= []
        format_scores =[]
        total_rewards = []
        for i, inp in enumerate(inputs): #多个问题 
            eval_refine_contents= []
            eval_prev_contents = []
            cur_format_scores = []
            cur_refine_answers= refined_answers[i * num_pre_Q:(i + 1) * num_pre_Q]
            cur_prev_answers = prev_answers[i * num_pre_Q:(i + 1) * num_pre_Q]
            # for j in range(i * num_pre_Q, (i + 1) * num_pre_Q): #一个问题的多个答案
            for j in range(len(cur_refine_answers)):
                format_score = reward_format(cur_refine_answers[j])
                cur_format_scores.append(format_score)
                eval_refine_contents.append({"q":inp['Q'],'std':inp['A'],"answer":cur_refine_answers[j],"sys_prompt":sys_prompt})
                eval_prev_contents.append({"q":inp['Q'],'std':inp['A'],"answer":cur_prev_answers[j], "sys_prompt": sys_prompt})
            #计算refine答案以及prev acc的得分
            # print(f"!!!type:{eval_refine_contents} {eval_refine_contents[0]}")
            # print(f"!!!type:{eval_prev_contents} {eval_prev_contents[1]}")

            # cur_acc_scores= requests.post(url=f"{eval_query_server}/generate", json = eval_refine_contents).json()
            # cur_prev_acc_scores = requests.post(url=f"{eval_query_server}/generate", json = eval_prev_contents).json()

            cur_acc_scores = [llm_eval(x['answer'], x) for x in eval_refine_contents]
            cur_prev_acc_scores = [llm_eval(x['answer'], x) for x in eval_prev_contents]

            # 计算当前refine答案的refine的得分
            print(f"prev_answers:{len(cur_prev_answers)} refined_answers:{len(cur_refine_answers)} cur_acc_scores:{len(cur_acc_scores)} cur_prev_acc_scores:{len(cur_prev_acc_scores)}")
            cur_refine_scores = reward_refine(cur_prev_answers,cur_refine_answers, cur_acc_scores, cur_prev_acc_scores)
            #奖励得分有三部分组成相加
            rewards = list(map(sum, zip(cur_format_scores, cur_acc_scores, cur_refine_scores)))
            #记录下来这一组答案的得分情况：
            for j in range(len(cur_refine_answers)):
                record_gen.append({"question": inp, "refined_answer": cur_refine_answers[j], "prev_answer":cur_prev_answers[j], "prev_score": cur_prev_acc_scores[j], "acc_score":cur_acc_scores[j], "format_score": cur_format_scores[j], "refine_score":cur_refine_scores[j], "reward_all":rewards[j]})
            total_rewards.extend(rewards)
            format_scores.extend(cur_format_scores)
            acc_scores.extend(cur_acc_scores)
        #record the generation data the score
        if os.path.exists(gen_data_path) and os.path.getsize(gen_data_path) > 0:
            with open(gen_data_path, 'r') as f:
                try:
                    gen_data = json.load(f)
                except json.JSONDecodeError:
                    gen_data = [] 
        else:
            gen_data = []  
        gen_data.extend(record_gen)
        with open(gen_data_path, 'w') as file:
            json.dump(gen_data, file, indent=4)

        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt_refine},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(total_rewards, dtype=torch.float32), refined_answers, torch.tensor(acc_scores, dtype=torch.float32), torch.tensor(format_scores, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence

    fout = open(f'{record_path}', 'w')
    for it in range(999999999):
       #按照顺序的方式来训练模型
        # if it==0:
        #     start = 0
        # else:
        #     start = 1000

        start = 0
        for j in range(start,len(QAs), Q_batch_size):
            inputs = QAs[j:j+Q_batch_size]
            if j % 2 == 0: 
                try_update_model()
            # inputs = random.sample(QAs, Q_batch_size)
            tic = time.time()
            prompt_inputs, rewards, answers, acc_scores, format_scores, scores = gen_samples(inputs)
            # print(f'time: {time.time()-tic:.2f}s    ', 'scores:', scores)
            fout.write(str(scores) + '\n')

            if it % 5 == 0: 
                fout.write(str(inputs[0])+"\n"+str(answers[0]) + '\n\n')
                fout.flush()
                # print('answers:', answers[0])
            ans_token_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)['input_ids']
            for i, pp in enumerate(prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_acc_scores = acc_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_format_scores = format_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                # pdb.set_trace()
                if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

                if ref_server_ver == 'tensor':
                    curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                    for ii in range(0, num_pre_Q, train_batch_size):
                        sub_rewards = curr_rewards[ii:ii+train_batch_size]
                        sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                        sub_acc_scores = curr_acc_scores[ii:ii+train_batch_size]
                        sub_format_scores = curr_format_scores[ii:ii+train_batch_size]

                        tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                        Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                        merged_ids = torch.cat([Qrep, output_ids], dim=1)
                        data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                        if compute_gen_logps:
                            zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                            zz = [ xx.prompt_logprobs[plen:] if xx.prompt_logprobs is not None else [] for xx in zz]
                            # zz = [xx.prompt_logprobs[plen:] for xx in zz]
                            if not zz:
                                print("[!!! SPEICIAL CASE]")
                                continue
                            gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                            data.append(tensor_to_bytes(gen_logps))
                        
                        data.append(tensor_to_bytes(sub_acc_scores))
                        data.append(tensor_to_bytes(sub_format_scores))
                        # print("!!data length:", len(data))
                        xdata = make_bytes_list(data)
                        # print("!!start to upload")
                        r = requests.post(f"{ref_server}/upload", data=xdata)
                        if r.content == b'string': ref_server_ver = 'string'
                elif ref_server_ver == 'string':
                    xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                            tensor_to_bytes(curr_rewards)])
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()
    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    float_params = [p for p in model.parameters() if p.requires_grad and p.dtype.is_floating_point]
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=float_params)

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: 
        progress = tqdm(progress)

    total_output_length = 0
    total_acc_correct = 0
    total_format_correct = 0
    total_num = 0

    # wandb.login(key=wandb_key)
    # wandb.init(project=wandb_project, name=wandb_name)
    
    for step in progress:
        batch = get_batch()
        while batch is None:
            time.sleep(1)
            batch = get_batch()

        # if batch['inputs'].shape[1]>2200:
        #     continue
        if torch.distributed.get_rank() == 0:
            batch_length = (batch['gen_logps'].shape[0] * batch['gen_logps'].shape[1])
            total_output_length += batch_length

            total_acc_correct += ( batch['acc_scores'] > 0).sum().item()
            total_format_correct += ( batch['format_scores'] > 0).sum().item()

            total_num += batch['inputs'].shape[0]
            # wandb.log({"avg_output_token_lenght": float(total_output_length / total_num),
            #             "acc_correct_ratio": float(total_acc_correct / total_num),
            #             "format_correct_ratio": float(total_format_correct / total_num),
            #          })
        loss = GRPO_step(batch)
        loss = loss.to(torch.float32) 
        engine.backward(loss)
        # print(f"!!!!rank:{torch.distributed.get_rank()} backward successfully ")
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
                update_model_num += 1
                print('!!The number of update the genmodel:',update_model_num)
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"{save_path}/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()