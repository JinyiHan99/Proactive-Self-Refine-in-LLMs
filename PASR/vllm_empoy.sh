CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /data/hjy/models/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port  8858\
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max_model_len 8000 \
    --tensor_parallel_size 1