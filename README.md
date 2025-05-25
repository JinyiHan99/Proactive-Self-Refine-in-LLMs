# A Stitch in Time Saves Nine: Proactive In-Process Self-Refinement for Language Models
[[📄 Paper (PDF)]](./figs/PASR_0516_final.pdf) 
- We propose PASR, a method that enables proactive self-refinement throughout the generation process via reinforcement learning.
- We design a comparison-based reward strategy to assess the effectiveness of proactive self-refinement and guide model behavior during training.
- We empirically demonstrate the effectiveness and efficiency of PASR across a diverse set of tasks. In particular, on Qwen3-8B, PASR significantly reduces average token consumption by 41.6\% compared to the standard generation method, while also achieving a 8.2\% improvement in accuracy.

<div style="text-align: center;">
    <img src="./figs/intro.jpeg" alt="intro" width = 600/>
    <p style="text-align: center;"><em>Figure 1:</em> Comparison between the post-hoc refinement method (left) and our proposed PASR (right). The post-hoc refinement method iteratively refines its initial answer. In contrast, PASR proactively refines its reasoning process during the generation.<br></p>
</div>

## Install Dependencies

```
conda create -n PASR python=3.10.9
conda activate PASR
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Run

```
# Step into the project directory
cd PASR

# Start the evaluation model to evaluate the quality of training rollout
# You can also use an API key instead
CUDA_VISIBLE_DEVICES=7 python eval_llm_server.py

# Start the GRPO refinement server model
CUDA_VISIBLE_DEVICES=0 python ref_server.py

# Start the PASR-GRPO training with DeepSpeed (using GPUs 1 and 2)
CUDA_VISIBLE_DEVICES=1,2 deepspeed pasr_grpo.py
```

You can configure the training parameters in ``PASR/config.py``. 

| Parameter     | Description                                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model_path`  | Path to the base language model (e.g., `"Qwen2.5-7B/"`). This is the model you fine-tune or refine during training.                                                      |
| `data_path`   | Path to the training dataset.  
| `gen_device`   | GPU ID used specifically for generation (e.g., 5), often different from training GPU.   |
| `Q_batch_size` | Number of generations per batch during rollout.                                         |
| `num_pre_Q`    | Group size for candidate generations (e.g., 8), used in candidate filtering or scoring. |     
| `eval_prompt` | Prompt template used for evaluation. It contains a question, ground truth answer, and AI-generated answer, and asks for binary scoring (1 for correct, 0 for incorrect). |
| `train_batch_size`  | Batch size for training steps (e.g., 2).                                                                                  |
| `all_steps`         | Total number of training steps (e.g., 3000).                                                                              |
| `ref_server` | URL of the reference model server used for evaluation or scoring. |
| `port`       | Port number for the local reference server (default is 59809).    |
| `wandb_key`     | Your wandb API key for authentication (keep this private). |


Notes:
- Ensure gen_device does not conflict with the GPU used for training if running both simultaneously.
- Use wandb to monitor training progress and generation quality in real time.

## Model Evaluation
We evaluate the model's performance using [VLLM](https://github.com/vllm-project/vllm) for fast and efficient inference. The test set used for evaluation is located at `/dataset/test_dataset`

We use the following generation configuration during evaluation:

```
sampling_params_stop = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=1500
)
```