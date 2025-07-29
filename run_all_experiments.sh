#!/bin/bash
# 批量提交所有实验的脚本

echo "开始提交所有实验到集群..."

# 提交 Single Agent 实验
echo "提交 Single Agent 实验..."
job1=$(sbatch run_single_agent.sh | awk '{print $4}')
echo "Single Agent 作业ID: $job1"

# 提交 Multi Agent 实验
echo "提交 Multi Agent 实验..."
job2=$(sbatch run_multi_agent.sh | awk '{print $4}')
echo "Multi Agent 作业ID: $job2"

# 提交 Intent Enhanced 实验
echo "提交 Intent Enhanced 实验..."
job3=$(sbatch run_intent_enhanced.sh | awk '{print $4}')
echo "Intent Enhanced 作业ID: $job3"

# 提交 Full Evidence 实验
echo "提交 Full Evidence 实验..."
job4=$(sbatch run_full_evidence.sh | awk '{print $4}')
echo "Full Evidence 作业ID: $job4"

# 提交 Multi People 实验（使用现有的脚本）
echo "提交 Multi People 实验..."
job5=$(sbatch run_multi_people.sh | awk '{print $4}')
echo "Multi People 作业ID: $job5"

echo "所有实验已提交！"
echo "作业ID列表:"
echo "  Single Agent: $job1"
echo "  Multi Agent: $job2"
echo "  Intent Enhanced: $job3"
echo "  Full Evidence: $job4"
echo "  Multi People: $job5"

echo "使用以下命令查看作业状态:"
echo "  squeue -u $USER"
echo "使用以下命令查看特定作业:"
echo "  scontrol show job <job_id>" 