#!/bin/bash
# 批量提交所有6个实验的脚本 - 智能版本

echo "=================================="
echo "Multi-Agent Debate 实验批量提交"
echo "=================================="
echo ""

# 检查当前GPU资源状态
echo "检查当前GPU资源状态..."
echo "可用的H100-96GB节点:"
sinfo -p gpu -o "%N %G %T" | grep "h100-96" | grep "idle"

echo ""
echo "可用的A100-80GB节点:"
sinfo -p gpu -o "%N %G %T" | grep "a100-80" | grep "idle"

echo ""
echo "=================================="
echo "开始提交所有6个实验到集群..."
echo "=================================="

# 提交 Single Agent + Basic Search
echo "1. 提交 Single Agent + Basic Search..."
job1=$(sbatch sbatch_single_basic.sh | awk '{print $4}')
echo "   作业ID: $job1"

# 等待一下再提交下一个，避免资源冲突
sleep 2

# 提交 Multi Agent + Basic Search
echo "2. 提交 Multi Agent + Basic Search..."
job2=$(sbatch sbatch_multi_basic.sh | awk '{print $4}')
echo "   作业ID: $job2"

sleep 2

# 提交 Single Agent + Intent Enhanced
echo "3. 提交 Single Agent + Intent Enhanced..."
job3=$(sbatch sbatch_single_intent.sh | awk '{print $4}')
echo "   作业ID: $job3"

sleep 2

# 提交 Multi Agent + Intent Enhanced
echo "4. 提交 Multi Agent + Intent Enhanced..."
job4=$(sbatch sbatch_multi_intent.sh | awk '{print $4}')
echo "   作业ID: $job4"

sleep 2

# 提交 Single Agent + Full Evidence
echo "5. 提交 Single Agent + Full Evidence..."
job5=$(sbatch sbatch_single_full.sh | awk '{print $4}')
echo "   作业ID: $job5"

sleep 2

# 提交 Multi Agent + Full Evidence
echo "6. 提交 Multi Agent + Full Evidence..."
job6=$(sbatch sbatch_multi_full.sh | awk '{print $4}')
echo "   作业ID: $job6"

echo "=================================="
echo "所有6个实验已提交！"
echo "=================================="
echo ""
echo "📋 作业ID列表:"
echo "  1. Single Agent + Basic Search:     $job1"
echo "  2. Multi Agent + Basic Search:      $job2"
echo "  3. Single Agent + Intent Enhanced:  $job3"
echo "  4. Multi Agent + Intent Enhanced:   $job4"
echo "  5. Single Agent + Full Evidence:    $job5"
echo "  6. Multi Agent + Full Evidence:     $job6"
echo ""
echo "🔧 资源配置:"
echo "  - GPU: H100-96GB (独占使用)"
echo "  - 内存: 64GB"
echo "  - CPU: 8核"
echo "  - 运行时间: Single模式12小时，Multi模式24小时"
echo ""
echo "📊 监控命令:"
echo "  查看所有作业状态:"
echo "    squeue -u $USER"
echo ""
echo "  查看特定作业详情:"
echo "    scontrol show job <job_id>"
echo ""
echo "  查看实时日志:"
echo "    tail -f logs/single_basic_<job_id>.out"
echo "    tail -f logs/multi_intent_<job_id>.out"
echo "    # 等等..."
echo ""
echo "  取消作业:"
echo "    scancel <job_id>"
echo ""
echo "✅ 所有脚本已配置为独占使用H100-96GB GPU，确保最佳性能！" 