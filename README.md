<h1 align="center">A Stitch in Time Saves Nine: Proactive In-Process Self-Refinement for Language Models</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-PDF-b5212f.svg?logo=adobe-acrobat-reader&logoColor=white)](./figs/PASR.pdf)
</div>

## 📢 Latest News

- **2026-02**: Our paper "A Stitch in Time Saves Nine: Proactive In-Process Self-Refinement for Language Models" has been accepted by **ICLR 2026** ! 🎉

## 📌 Overview

We propose **PASR** (Proactive In-Process Self-Refinement), a novel method that enables language models to proactively refine their outputs *during* the generation process via reinforcement learning. Unlike traditional post-hoc refinement approaches, PASR intervenes early to correct errors as they emerge, significantly improving efficiency and quality.

### Key Contributions

- **Proactive Refinement Strategy**: Enables self-refinement throughout the generation process rather than just at the end
- **Comparison-Based Reward Function**: Designing a novel reward mechanism to assess refinement effectiveness and guide training
- **Empirical Effectiveness**: Demonstrating significant improvements across diverse tasks. On Qwen3-8B:
  - **41.6% reduction** in average token consumption
  - **8.2% improvement** in accuracy

<div align="center">
    <img src="./figs/pasr_intro.jpg" alt="PASR Introduction" width="600"/>
</div>

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (recommended: ≥16GB VRAM)
- Python 3.10+
- PyTorch 2.5+

### Installation

```bash
# Create conda environment
conda create -n PASR python=3.10.9
conda activate PASR

# Install PyTorch (CUDA 12.4)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Training

```bash
cd PASR

# Step 1: Start the evaluation model server for rollout quality assessment
bash vllm_empoy.sh

# Step 2: Start the GRPO refinement server
CUDA_VISIBLE_DEVICES=1 python ref_server.py

# Step 3: Start PASR-GRPO training with DeepSpeed
CUDA_VISIBLE_DEVICES=2,3 deepspeed pasr_main.py
```

### Configuration

All training parameters are configured in `PASR/config.py`. Key parameters include:

| Parameter          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `model_path`      | Path to the base language model (e.g., `"Qwen2.5-7B/"`)                     |
| `data_path`       | Path to the training dataset                                                |
| `gen_device`      | GPU ID for generation (separate from training GPU)                          |
| `Q_batch_size`    | Number of generations per batch during rollout                              |
| `num_pre_Q`       | Group size for candidate generations (used in scoring/filtering)            |
| `eval_prompt`     | Prompt template for binary evaluation (1 for correct, 0 for incorrect)      |
| `train_batch_size`| Training batch size (e.g., 2)                                               |
| `all_steps`       | Total training steps (e.g., 3000)                                           |
| `ref_server`      | URL of the reference model server for evaluation                            |
| `port`            | Port number for local reference server (default: 59809)                      |
| `wandb_key`       | Weights & Biases API key for experiment tracking (keep private)             |

**Notes**:
- Ensure `gen_device` does not conflict with training GPU if running simultaneously
- Use Weights & Biases to monitor training progress and generation quality in real-time

## 📊 Evaluation

### Inference

```bash
cd eval

# Run inference on trained model
CUDA_VISIBLE_DEVICES=1 python inference.py \
  --model_name your_checkpoint \
  --data_names dataset1 dataset2 \
  --output_path ./eval_results

# Evaluate using VLLM
python eval_with_vllm

# Calculate final average scores
python get_scores.py
```

### Evaluation Configuration

We use the following generation parameters during evaluation:

```python
sampling_params_stop = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=1500
)
```

## 📁 Data

We provide comprehensive training and test datasets in the `/data` directory. The datasets cover diverse task types to ensure robust evaluation.

## 📚 References

```bibtex
@misc{han2025stitchtimesavesnine,
      title={A Stitch in Time Saves Nine: Proactive Self-Refinement for Language Models}, 
      author={Jinyi Han and Xinyi Wang and Haiquan Zhao and Tingyun li and Zishang Jiang and Sihang Jiang and Jiaqing Liang and Xin Lin and Weikang Zhou and Zeye Sun and Fei Yu and Yanghua Xiao},
      year={2025},
      eprint={2508.12903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.12903}, 
}
```
