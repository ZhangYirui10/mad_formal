#!/bin/bash
#SBATCH --job-name=mad_multi_intent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_intent_%j.out
#SBATCH --error=logs/multi_intent_%j.err
#SBATCH --exclusive

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
echo "Experiment: Multi Agent + Intent Enhanced"
echo "GPU Type: H100-96GB (Exclusive)"

# 运行主程序 - Multi Agent + Intent Enhanced
python main.py \
    --mode multi \
    --input_file data/intent_enhanced_con_pro_bge_large_400_top20_by_score_with_evi.json

echo "Job completed at $(date)" 