#!/bin/bash
#SBATCH --job-name=mad_multi_people
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_people_%j.out
#SBATCH --error=logs/multi_people_%j.err

# 创建日志目录
mkdir -p logs

# 加载必要的模块（根据你的集群配置调整）
module load cuda/11.8
module load python/3.9

# 激活conda环境（如果你使用conda）
# source activate mad_debate

# 或者如果你使用venv
# source venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 运行主程序
python main.py \
    --mode multi_people \
    --input_file /Users/yiruizhang/Desktop/mad_formal/data/intent_enhanced_con_pro_bge_large_400_top20_by_score_with_evi.json

echo "Job completed at $(date)" 