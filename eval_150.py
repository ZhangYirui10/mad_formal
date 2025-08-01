import os
import json
import re
import sys
import argparse
from collections import defaultdict

def extract_verdict_single(verdict_list):
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
    # Try to get verdict from different possible field names
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

def determine_mode(sample_value):
    if isinstance(sample_value, list):
        return "single"
    elif isinstance(sample_value, dict) and ("final_verdict" in sample_value or "verdict" in sample_value):
        return "multi"
    else:
        return "multi"

def convert_prediction_file(pred_file):
    with open(pred_file, "r") as f:
        data = json.load(f)
    first_item = next(iter(data.values()))
    mode = determine_mode(first_item)

    id_to_verdict = {}
    for example_id, content in data.items():
        if mode == "single":
            verdict = extract_verdict_single(content)
        else:  # multi
            verdict = extract_verdict_multi(content)
        id_to_verdict[example_id] = verdict
    return id_to_verdict, mode

def get_150_sample_ids():
    """Get 150 sample IDs from veracity_examples_results.json"""
    with open("veracity_examples_results.json", "r") as f:
        data = json.load(f)
    
    all_ids = []
    for category, ids in data.items():
        all_ids.extend(ids)
    
    # Convert to strings to match prediction file keys
    return set(str(id) for id in all_ids)

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def calculate_class_metrics(y_true, y_pred, class_label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == class_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_label and p == class_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_label and p != class_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1_score(precision, recall)
    return precision, recall, f1

def evaluate_150_samples(pred_file, groundtruth_file):
    """Evaluate 150 samples specifically"""
    prediction, mode = convert_prediction_file(pred_file)
    
    # Get 150 sample IDs
    sample_ids = get_150_sample_ids()
    
    with open(groundtruth_file, "r") as f:
        groundtruth = json.load(f)

    total, correct = 0, 0
    label_count = defaultdict(int)
    correct_count = defaultdict(int)
    y_true, y_pred = [], []
    
    # Count distribution of 150 samples
    sample_distribution = defaultdict(int)

    for key in groundtruth:
        if key in prediction and key in sample_ids:
            total += 1
            true_label = groundtruth[key]
            pred_label = prediction[key]
            y_true.append(true_label)
            y_pred.append(pred_label)
            label_count[true_label] += 1
            sample_distribution[true_label] += 1
            if true_label == pred_label:
                correct += 1
                correct_count[true_label] += 1

    accuracy = correct / total if total > 0 else 0.0
    metrics = {label: calculate_class_metrics(y_true, y_pred, label) for label in ["TRUE", "HALF-TRUE", "FALSE"]}
    macro_f1 = sum(f1 for _, _, f1 in metrics.values()) / 3

    print(f"File: {pred_file}")
    print(f"  Mode: {mode}")
    print(f"  150 Sample Evaluation Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall Accuracy: {accuracy:.2%}\n")
    
    print("  Sample Distribution:")
    for label in ["TRUE", "HALF-TRUE", "FALSE"]:
        count = sample_distribution[label]
        print(f"    {label}: {count} samples")
    
    print("\n  Class-wise Accuracy:")
    for label in ["TRUE", "HALF-TRUE", "FALSE"]:
        correct_num = correct_count[label]
        total_num = label_count[label]
        acc = correct_num / total_num if total_num > 0 else 0.0
        print(f"    {label}: {acc:.2%} ({correct_num}/{total_num})")
    
    print("\n  F1 Scores:")
    for label in ["TRUE", "HALF-TRUE", "FALSE"]:
        p, r, f1 = metrics[label]
        print(f"    {label} - Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1:.2%}")
    print(f"  Macro-F1: {macro_f1:.2%}")
    print("-" * 80)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': metrics,
        'sample_distribution': dict(sample_distribution)
    }

if __name__ == "__main__":
    # Fixed groundtruth file path
    groundtruth_file = "data/GT_test_all.json"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate 150 sample predictions')
    parser.add_argument('--prediction', nargs='+', required=True, 
                       help='Prediction file(s) to evaluate (can specify multiple files)')
    
    args = parser.parse_args()
    prediction_files = args.prediction
    
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"Prediction files: {prediction_files}")
    print("=" * 80)
    
    results = {}
    for pred_file in prediction_files:
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} does not exist, skipping...")
            continue
        result = evaluate_150_samples(pred_file, groundtruth_file)
        results[pred_file] = result
    
    # If multiple files, compare results
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Multi-file Comparison Results:")
        print("=" * 80)
        
        for pred_file, result in results.items():
            print(f"{pred_file}:")
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Macro-F1: {result['macro_f1']:.2%}")
            print() 