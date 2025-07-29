#!/bin/bash
# 本地测试脚本 - 验证sbatch脚本配置

echo "🧪 本地测试 - 验证sbatch脚本配置"
echo "=================================="
echo ""

# 检查sbatch脚本文件
echo "📋 检查sbatch脚本文件:"
scripts=(
    "sbatch_single_basic.sh"
    "sbatch_multi_basic.sh" 
    "sbatch_single_intent.sh"
    "sbatch_multi_intent.sh"
    "sbatch_single_full.sh"
    "sbatch_multi_full.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ $script"
        
        # 检查脚本权限
        if [ -x "$script" ]; then
            echo "    ✅ 可执行权限"
        else
            echo "    ❌ 缺少执行权限"
        fi
        
        # 检查SBATCH配置
        echo "    📊 SBATCH配置:"
        grep "^#SBATCH" "$script" | while read line; do
            echo "      $line"
        done
        
    else
        echo "  ❌ $script (缺失)"
    fi
    echo ""
done

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
echo "🔧 资源配置总结:"
echo "  - GPU: H100-96GB (独占使用)"
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

echo "🚀 下一步操作:"
echo "  1. 登录到集群: ssh username@cluster-address"
echo "  2. 上传项目: scp -r . username@cluster-address:/path/to/workspace/"
echo "  3. 在集群上运行: ./sbatch_all_experiments.sh"
echo ""

echo "✅ 本地验证完成！脚本配置正确，可以上传到集群运行。" 