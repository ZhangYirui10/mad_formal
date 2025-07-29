#!/bin/bash
#SBATCH --job-name=mad_intent_enhanced
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/intent_enhanced_%j.out
#SBATCH --error=logs/intent_enhanced_%j.err

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

# 运行主程序 - Intent Enhanced模式
python main.py \
    --mode multi \
    --input_file data/intent_enhanced_con_pro_bge_large_400_top20_by_score_with_evi.json

echo "Job completed at $(date)" 