CUDA_VISIBLE_DEVICES=4 nohup python infer_with_oracle.py --model_name /mnt/data/kw/models/Qwen2.5-14B --data_names math gsm8k aime24 --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/vallina > ./logs/0730/Qwen2.5-14B_vallina_with_oracle_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python infer_with_oracle.py --model_name /mnt/data/kw/models/Qwen2.5-14B --data_names mmlu drop --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/vallina > ./logs/0730/Qwen2.5-14B_vallina_with_oracle_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python infer_with_oracle.py --model_name /mnt/data/kw/models/Qwen2.5-14B --data_names xsum arc-challenge gpqa --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/vallina > ./logs/0730/Qwen2.5-14B_vallina_with_oracle_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python infer_with_oracle.py --model_name /mnt/data/kw/models/Qwen2.5-14B --data_names wino commonsenseqa --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/vallina > ./logs/0730/Qwen2.5-14B_vallina_with_oracle_3.log 2>&1 &
