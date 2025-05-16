# export CUDA_VISIBLE_DEVICES=2\
cd /data2/WangXinyi/refine/main
conda activate hqx_grpo
nohup env CUDA_VISIBLE_DEVICES=7 python ref_server.py > ./logs/0320/server.log 2>&1 & 
nohup deepspeed --include=localhost:5,6 --master_port 12989 hjy_grpo_vllm_one.py > ./logs/0320/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 screen -L -Logfile ./logs/0406/server_v1.log python ref_server.py
CUDA_VISIBLE_DEVICES=3,4 screen -L -Logfile ./logs/0406/run_v1.log deepspeed grpo_vllm_one_refine.py 


CUDA_VISIBLE_DEVICES=4,5 screen -L -Logfile ./logs/0329/reward_model_1.log python -m vllm.entrypoints.openai.api_server \
  --model /data2/Qwen/Qwen2.5-14B-Instruct/ \
  --port 8853 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --disable-log-requests

CUDA_VISIBLE_DEVICES=0,1,6,7 screen -L -Logfile ./logs/0406/reward_model_1.log python -m vllm.entrypoints.openai.api_server \
  --model /data2/Qwen/Qwen2.5-32B-Instruct/ \
  --port 8858 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --disable-log-requests
