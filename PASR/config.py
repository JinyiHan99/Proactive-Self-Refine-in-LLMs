train_config = {
    "train_batch_size":2,
    "all_steps": 3000,
    "save_steps": 300,
    "wandb_name":"0406_v1",
    "wandb_project":"refine",
    "save_path":  "./ckp/qwen-14b-refine",
    "record_path": "./reward_record_v1.txt",
    "gen_data_path": "./gen_data_v1.json",
    "gen_device":5,  
    "data_path":"data/alpaca_evol_instruct_70k_select.json",
    "beta": 0.04,
    "model_path": "Qwen2.5-14B/",
    "Q_batch_size": 5,
    "num_pre_Q": 8,
    "train_batch_size":2,
    "gen_update_steps": 16,
    "compute_gen_logps": True,
    "clip_param": 0.2,
    "ref_server": "http://localhost:59809",
    "port": 59809,
    "wandb_key":""
}

ds_config = {
    "train_micro_batch_size_per_gpu": train_config['train_batch_size'],
    "gradient_accumulation_steps": 4,
    "steps_per_print": 5,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}