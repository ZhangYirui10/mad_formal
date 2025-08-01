#!/bin/bash
#SBATCH --job-name=llama_four_agents_intent
#SBATCH --output=logs/llama_four_agents_intent_%j.out
#SBATCH --error=logs/llama_four_agents_intent_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

# 初始化环境
source ~/.bashrc
conda activate mad_debate

# 创建日志目录
mkdir -p logs

# 执行命令
python -u main.py --mode four_agents_intent --model llama --input_file /data/matching_evidence.json 