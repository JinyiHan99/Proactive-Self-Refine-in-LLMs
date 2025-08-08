# CUDA_VISIBLE_DEVICES=0 nohup python infer_self_refine_first_answer.py --data_names math gsm8k > ./logs/0730/PTR_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python infer_self_refine_first_answer.py --data_names aime24 mmlu > ./logs/0730/PTR_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python infer_self_refine_first_answer.py --data_names arc-challenge > ./logs/0730/PTR_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python infer_self_refine_first_answer.py --data_names drop wino > ./logs/0730/PTR_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python infer_self_refine_first_answer.py --data_names xsum > ./logs/0730/PTR_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python infer_self_refine_first_answer.py --data_names gpqa > ./logs/0730/PTR_5.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python infer_self_refine_first_answer.py --data_names commonsenseqa > ./logs/0730/PTR_6.log 2>&1 &
