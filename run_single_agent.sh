#!/bin/bash
#SBATCH --job-name=mad_single_agent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/single_agent_%j.out
#SBATCH --error=logs/single_agent_%j.err

# 创建日志目录
mkdir -p logs

# 加载必要的模块（根据你的集群配置调整）
module load cuda/11.8
module load python/3.13.2

# 激活conda环境
source activate mad_debate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# 运行主程序 - Single Agent模式
python main.py \
    --mode single \
    --input_file data/retrieved_evidence_bgebase.json

echo "Job completed at $(date)" 