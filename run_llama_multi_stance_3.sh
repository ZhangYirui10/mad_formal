#!/bin/bash
#SBATCH --job-name=llama_stance_3
#SBATCH --output=logs/llama_stance_3_%j.out
#SBATCH --error=logs/llama_stance_3_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-long                # 更容易抢到资源的分区
#SBATCH --gres=gpu:h100-47:1                # 使用集群上可用的 h100-47:4 配置
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

# 加载环境
source ~/.bashrc
conda activate mad_debate

# 创建日志目录
mkdir -p logs

# 执行任务 - 使用llama模型进行3立场辩论
python main.py --mode multi_stance_3 --model llama --input_file data/matching_evidence.json 