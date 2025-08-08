# nohup python eval_with_llm_without_oracle.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/without_oracle > ./logs/0730/eval_without_oracle.log 2>&1
# nohup python eval_with_llm.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/STaR > ./logs/0730/eval_star.log 2>&1
# nohup python eval_with_llm.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/SCoRE > ./logs/0730/eval_score.log 2>&1 
# nohup python eval_with_llm.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/PASR_IFT > ./logs/0730/eval_pasr_ift.log 2>&1 
# nohup python eval_with_llm_2.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/ISC > ./logs/0730/eval_isc.log 2>&1 
nohup python eval_with_llm.py --main_data_path /mnt/data/kw/wxy/self_refine/Qwen2.5-14B/PTR > ./logs/0730/eval_ptr.log 2>&1 

