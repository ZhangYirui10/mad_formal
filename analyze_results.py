import json
import re

def extract_verdict_single(verdict_list):
    """从single格式中提取verdict"""
    text = verdict_list[0] if verdict_list else ""
    patterns = [
        r'\[?VERDICT\]:\s*(TRUE|FALSE|HALF-TRUE)',
        r'\*\*VERDICT\*\*:.*?(TRUE|FALSE|HALF-TRUE)',
        r'VERDICT\s*:\s*(TRUE|FALSE|HALF-TRUE)',
        r'VERDICT.*?(TRUE|FALSE|HALF-TRUE)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "UNKNOWN"

def extract_verdict_multi(content):
    """从multi格式中提取verdict"""
    text = content.get("final_verdict", "") or content.get("verdict", "")
    patterns = [
        r'\[?VERDICT\]:\s*(TRUE|FALSE|HALF-TRUE)',
        r'VERDICT\s*:\s*(TRUE|FALSE|HALF-TRUE)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).upper()
    keyword_match = re.search(r'\b(TRUE|FALSE|HALF-TRUE)\b', text, re.IGNORECASE)
    return keyword_match.group(1).upper() if keyword_match else "UNKNOWN"

def load_predictions(file_path):
    """加载预测文件并提取verdict"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for example_id, content in data.items():
        if isinstance(content, list):
            verdict = extract_verdict_single(content)
        else:
            verdict = extract_verdict_multi(content)
        predictions[example_id] = verdict
    
    return predictions

def main():
    # 加载groundtruth
    with open('data/GT_test_all.json', 'r') as f:
        groundtruth = json.load(f)
    
    # 加载三个预测文件
    file1_predictions = load_predictions('data_compare/retrieved_evidence_bgebase_answer_map_multi_people.json')
    file2_predictions = load_predictions('data_compare/retrieved_evidence_bgebase_answer_map_multi.json')
    file3_predictions = load_predictions('data_compare/retrieved_evidence_bgebase_answer_map_single.json')
    
    # 找出第一个文件判断正确，后两个文件判断错误的案例
    correct_cases = []
    
    for example_id, true_label in groundtruth.items():
        # 检查三个文件是否都有结果
        if (example_id in file1_predictions and 
            example_id in file2_predictions and 
            example_id in file3_predictions):
            
            file1_pred = file1_predictions[example_id]
            file2_pred = file2_predictions[example_id]
            file3_pred = file3_predictions[example_id]
            
            # 检查第一个文件判断正确，后两个文件判断错误
            if (file1_pred == true_label and 
                file2_pred != true_label and 
                file3_pred != true_label):
                
                correct_cases.append({
                    'example_id': example_id,
                    'groundtruth': true_label,
                    'file1_prediction': file1_pred,
                    'file2_prediction': file2_pred,
                    'file3_prediction': file3_pred
                })
    
    # 输出结果
    print(f"找到 {len(correct_cases)} 个案例：第一个文件判断正确，后两个文件判断错误")
    print("=" * 80)
    
    for case in correct_cases:
        print(f"案例ID: {case['example_id']}")
        print(f"Groundtruth: {case['groundtruth']}")
        print(f"文件1预测: {case['file1_prediction']} ✓")
        print(f"文件2预测: {case['file2_prediction']} ✗")
        print(f"文件3预测: {case['file3_prediction']} ✗")
        print("-" * 40)
    
    # 统计不同错误类型的分布
    print("\n错误类型统计:")
    error_types = {}
    for case in correct_cases:
        file2_error = f"{case['groundtruth']} -> {case['file2_prediction']}"
        file3_error = f"{case['groundtruth']} -> {case['file3_prediction']}"
        
        error_types[file2_error] = error_types.get(file2_error, 0) + 1
        error_types[file3_error] = error_types.get(file3_error, 0) + 1
    
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{error_type}: {count}次")

if __name__ == "__main__":
    main() 