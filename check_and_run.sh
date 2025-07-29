#!/bin/bash
# 检查和运行脚本

echo "🚀 Multi-Agent Debate 实验运行指南"
echo "=================================="
echo ""

# 检查当前目录
echo "📁 当前工作目录: $(pwd)"
echo ""

# 检查脚本文件是否存在
echo "📋 检查脚本文件:"
scripts=(
    "sbatch_single_basic.sh"
    "sbatch_multi_basic.sh" 
    "sbatch_single_intent.sh"
    "sbatch_multi_intent.sh"
    "sbatch_single_full.sh"
    "sbatch_multi_full.sh"
    "sbatch_all_experiments.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script (缺失)"
    fi
done

echo ""
echo "🔍 检查数据文件:"
data_files=(
    "data/retrieved_evidence_bgebase.json"
    "data/intent_enhanced_con_pro_bge_large_400_top20_by_score_with_evi.json"
    "data/full_evidence.json"
)

for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file (缺失)"
    fi
done

echo ""
echo "🔧 资源配置说明:"
echo "  - GPU: H100-96GB (独占使用，每个作业独立GPU)"
echo "  - 内存: 64GB"
echo "  - CPU: 8核"
echo "  - 运行时间: Single模式12小时，Multi模式24小时"
echo ""

echo "📊 实验配置:"
echo "  1. Single Agent + Basic Search"
echo "  2. Multi Agent + Basic Search"
echo "  3. Single Agent + Intent Enhanced"
echo "  4. Multi Agent + Intent Enhanced"
echo "  5. Single Agent + Full Evidence"
echo "  6. Multi Agent + Full Evidence"
echo ""

echo "🎯 运行方式:"
echo ""
echo "方式1: 一键提交所有6个实验"
echo "  ./sbatch_all_experiments.sh"
echo ""
echo "方式2: 单独提交某个实验"
echo "  sbatch sbatch_single_basic.sh    # Single Agent + Basic"
echo "  sbatch sbatch_multi_intent.sh    # Multi Agent + Intent"
echo "  # 等等..."
echo ""
echo "方式3: 检查GPU资源后再提交"
echo "  sinfo -p gpu -o \"%N %G %T\" | grep \"h100-96\""
echo "  ./sbatch_all_experiments.sh"
echo ""

echo "📈 监控命令:"
echo "  squeue -u \$USER                    # 查看所有作业"
echo "  scontrol show job <job_id>         # 查看特定作业"
echo "  tail -f logs/single_basic_<job_id>.out  # 查看日志"
echo "  scancel <job_id>                   # 取消作业"
echo ""

echo "⚠️  重要提醒:"
echo "  - 每个作业都会独占一个H100-96GB GPU"
echo "  - 确保有足够的GPU资源再提交"
echo "  - 建议先检查GPU可用性: sinfo -p gpu"
echo "  - 如果资源不足，可以分批提交"
echo ""

echo "✅ 准备就绪！选择运行方式开始实验。" 