#!/bin/bash
#SBATCH --job-name=llama_multi_people_4r
#SBATCH --output=logs/llama_multi_people_4r_%j.out
#SBATCH --error=logs/llama_multi_people_4r_%j.err
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
python -u main.py --mode multi_people_4r --model qwen --input_file data/extracted_retrieved_evidence.json