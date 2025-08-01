#!/usr/bin/env python3
"""
提取在veracity_examples_results.json中出现的编号对应的retrieved_evidence_bgebase.json内容
"""

import json
import os

def extract_matching_evidence():
    """
    从retrieved_evidence_bgebase.json中提取在veracity_examples_results.json中出现的编号内容
    """
    
    # 读取veracity_examples_results.json，获取所有编号
    print("正在读取veracity_examples_results.json...")
    with open('veracity_examples_results.json', 'r', encoding='utf-8') as f:
        veracity_data = json.load(f)
    
    # 收集所有编号（转换为字符串以便匹配）
    all_ids = set()
    for category, ids in veracity_data.items():
        all_ids.update(str(id_num) for id_num in ids)
    
    print(f"在veracity_examples_results.json中找到 {len(all_ids)} 个唯一编号")
    
    # 读取retrieved_evidence_bgebase.json
    print("正在读取retrieved_evidence_bgebase.json...")
    with open('data/retrieved_evidence_bgebase.json', 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)
    
    print(f"retrieved_evidence_bgebase.json包含 {len(evidence_data)} 个条目")
    
    # 提取匹配的内容
    matching_evidence = {}
    found_count = 0
    
    for evidence_id, evidence_content in evidence_data.items():
        if evidence_id in all_ids:
            matching_evidence[evidence_id] = evidence_content
            found_count += 1
    
    print(f"找到 {found_count} 个匹配的条目")
    
    # 保存结果
    output_file = 'matching_evidence.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matching_evidence, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_file}")
    
    # 按类别统计
    print("\n按类别统计:")
    for category, ids in veracity_data.items():
        category_count = sum(1 for evidence_id in ids if str(evidence_id) in matching_evidence)
        print(f"{category}: {category_count}/{len(ids)} 个条目")
    
    return matching_evidence

def main():
    """主函数"""
    try:
        matching_evidence = extract_matching_evidence()
        
        # 显示一些示例
        print("\n示例条目:")
        for i, (evidence_id, content) in enumerate(matching_evidence.items()):
            if i >= 3:  # 只显示前3个
                break
            print(f"\n编号 {evidence_id}:")
            print(f"Claim: {content.get('claim', 'N/A')}")
            print(f"Evidence数量: {len(content.get('evidence_full_text', []))}")
            
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e}")
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 {e}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 