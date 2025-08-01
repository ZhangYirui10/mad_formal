#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算指标脚本
为指定的预测文件计算准确率、F1分数等指标
"""

import json
import re
import sys
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report

def extract_final_verdict(text):
    """从文本中提取final_verdict"""
    if not text:
        return None
    
    # 查找标准的 [VERDICT]: 格式
    verdict_match = re.search(r'\[VERDICT\]:\s*([A-Z\s-]+)', text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip().lower()
        return normalize_verdict(verdict)
    
    # 查找其他格式
    text_lower = text.lower()
    
    # 检查half-true（优先级最高）
    if re.search(r'\bhalf\s*true\b', text_lower):
        return 'half-true'
    
    # 检查false
    if re.search(r'\bfalse\b', text_lower):
        return 'false'
    
    # 检查true
    if re.search(r'\btrue\b', text_lower):
        return 'true'
    
    return None

def normalize_verdict(verdict):
    """标准化判决结果"""
    if not verdict:
        return None
    
    verdict = verdict.lower().strip()
    
    # 标准化half-true
    if 'half' in verdict and 'true' in verdict:
        return 'half-true'
    elif 'partially' in verdict or 'partly' in verdict:
        return 'half-true'
    
    # 标准化true
    if verdict == 'true' or verdict == 'correct' or verdict == 'accurate':
        return 'true'
    
    # 标准化false
    if verdict == 'false' or verdict == 'incorrect' or verdict == 'wrong':
        return 'false'
    
    return verdict

def load_predictions(file_path):
    """加载预测结果"""
    predictions = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for example_id, result_dict in data.items():
            # 检查是否是字典格式（包含final_verdict字段）
            if isinstance(result_dict, dict) and 'final_verdict' in result_dict:
                result_text = result_dict['final_verdict']
                verdict = extract_final_verdict(result_text)
                if verdict:
                    predictions[example_id] = verdict
            # 检查是否是列表格式
            elif isinstance(result_dict, list) and len(result_dict) > 0:
                result_text = result_dict[0]
                verdict = extract_final_verdict(result_text)
                if verdict:
                    predictions[example_id] = verdict
                    
    except Exception as e:
        print(f"加载预测文件时出错: {e}")
        
    return predictions

def load_ground_truth(file_path, limit=None):
    """加载真实标签"""
    ground_truth = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if limit and count >= limit:
                break
                
            if 'example_id' in item and 'veracity' in item:
                example_id = str(item['example_id'])
                veracity = item['veracity'].lower()
                ground_truth[example_id] = veracity
                count += 1
                
    except Exception as e:
        print(f"加载真实标签文件时出错: {e}")
        
    return ground_truth

def calculate_metrics(predictions, ground_truth):
    """计算各种指标"""
    # 找到共同的example_id
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    
    if not common_ids:
        print("没有找到匹配的example_id")
        return None
    
    print(f"找到 {len(common_ids)} 个匹配的样本")
    
    # 准备数据
    y_true = []
    y_pred = []
    
    for example_id in sorted(common_ids):
        y_true.append(ground_truth[example_id])
        y_pred.append(predictions[example_id])
    
    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 计算每个类别的F1分数
    f1_true = f1_score(y_true, y_pred, average=None, labels=['true'])[0] if 'true' in y_true or 'true' in y_pred else 0
    f1_half_true = f1_score(y_true, y_pred, average=None, labels=['half-true'])[0] if 'half-true' in y_true or 'half-true' in y_pred else 0
    f1_false = f1_score(y_true, y_pred, average=None, labels=['false'])[0] if 'false' in y_true or 'false' in y_pred else 0
    
    return {
        'acc': acc,
        'macro_f1': macro_f1,
        'f1_true': f1_true,
        'f1_half_true': f1_half_true,
        'f1_false': f1_false,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    # 文件路径
    predictions_file = "/home/qqs/mad_formal1/data/full_evidence_answer_map_multi_people_3_qwen.json"
    
    ground_truth_files = [
        "/home/qqs/mad_formal1/data/test.json"
    ]
    
    print("=" * 60)
    print("指标计算报告")
    print("=" * 60)
    
    print("\n1. 数据加载")
    print("-" * 30)
    predictions = load_predictions(predictions_file)
    print(f"预测文件: 加载了 {len(predictions)} 个预测")
    print(f"预测文件路径: {predictions_file}")
    
    # 尝试加载ground truth
    ground_truth = None
    for gt_file in ground_truth_files:
        if os.path.exists(gt_file):
            print(f"尝试加载ground truth文件: {gt_file}")
            ground_truth = load_ground_truth(gt_file, limit=2000)
            if ground_truth:
                print(f"成功加载 {len(ground_truth)} 个真实标签")
                break
    
    if not ground_truth:
        print("无法找到有效的ground truth文件")
        return
    
    print("\n2. 指标计算")
    print("-" * 30)
    metrics = calculate_metrics(predictions, ground_truth)
    
    if metrics:
        print("\n3. 最终结果")
        print("-" * 30)
        print(f"Accuracy (准确率):     {metrics['acc']:.4f}")
        print(f"Macro F1 (宏平均F1):   {metrics['macro_f1']:.4f}")
        print(f"F1 (True):            {metrics['f1_true']:.4f}")
        print(f"F1 (Half-True):       {metrics['f1_half_true']:.4f}")
        print(f"F1 (False):           {metrics['f1_false']:.4f}")
        
        print("\n4. 详细分类报告")
        print("-" * 30)
        print(classification_report(metrics['y_true'], metrics['y_pred'], 
                                 labels=['true', 'half-true', 'false']))
        
        # 保存结果到文件
        result_file = "metrics_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': metrics['acc'],
                'macro_f1': metrics['macro_f1'],
                'f1_true': metrics['f1_true'],
                'f1_half_true': metrics['f1_half_true'],
                'f1_false': metrics['f1_false'],
                'total_samples': len(metrics['y_true'])
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {result_file}")
    else:
        print("无法计算指标")

if __name__ == "__main__":
    main() 