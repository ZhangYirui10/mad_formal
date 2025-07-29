import argparse
import json
from tqdm import tqdm
import os
from agents.single_agent import verify_claim
from agents.multi_agents import (
    opening_pro, rebuttal_pro, closing_pro,
    opening_con, rebuttal_con, closing_con,
    judge_final_verdict
)
import torch

def run_single_agent(claim, evidence):
    return verify_claim(claim, evidence)

def run_multi_agent(claim, evidence):
    print("\n=== Running Multi-Agent Debate (3 rounds) ===")
    pro_open = opening_pro(claim, evidence)
    con_open = opening_con(claim, evidence)
    pro_rebut = rebuttal_pro(claim, evidence, con_open)
    con_rebut = rebuttal_con(claim, evidence, pro_open)
    pro_close = closing_pro(claim, evidence)
    con_close = closing_con(claim, evidence)
    final_result = judge_final_verdict(
        claim, evidence,
        pro_open, con_open,
        pro_rebut, con_rebut,
        pro_close, con_close
    )
    return pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Choose inference mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing examples."
    )
    args = parser.parse_args()

    # Load input file
    print(f"Loading input file: {args.input_file}")
    with open(args.input_file, "r") as f:
        all_examples = json.load(f)
    
    # Generate output filename based on input filename
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file = os.path.join("data", f"{input_basename}_answer_map_{args.mode}.json")
    
    print(f"Output will be saved to: {output_file}")
    print(f"Processing {len(all_examples)} examples in {args.mode} mode")

    try:
        with open(output_file, "r") as f:
            answer_map = json.load(f)
    except FileNotFoundError:
        answer_map = {}

    for example_id, example in tqdm(all_examples.items(), desc=f"Processing examples ({args.mode})"):
        if example_id in answer_map:
            continue

        claim = example["claim"]
        # evidence = example["evidence"]
        evidence = example["evidence_full_text"]
        # evidence_pro_text = example["evidence_pro_text"]
        # evidence_con_text = example["evidence_con_text"]

        if args.mode == "single":
            result = run_single_agent(claim, evidence)
            answer_map[example_id] = [result]

        elif args.mode == "multi_role":
            # Step 1: 推理 intent 和角色
            intent, support_role, oppose_role = infer_intent_and_roles(claim)

            # Step 2: Opening statements
            pro_open = opening_pro(claim, evidence, role=support_role)
            con_open = opening_con(claim, evidence, role=oppose_role)

            # Step 3: Rebuttals
            # pro_rebut = rebuttal_pro(claim, evidence, con_open)
            # con_rebut = rebuttal_con(claim, evidence, pro_open)
            pro_rebut = rebuttal_pro(claim, evidence, con_open, role=support_role)
            con_rebut = rebuttal_con(claim, evidence, pro_open, role=oppose_role)

            # Step 4: Closings
            # pro_close = closing_pro(claim, evidence)
            # con_close = closing_con(claim, evidence)
            pro_close = closing_pro(claim, evidence, role=support_role)
            con_close = closing_con(claim, evidence, role=oppose_role)

            # Step 5: Judge verdict
            final_result = judge_final_verdict(
                claim, evidence,
                pro_open, con_open,
                pro_rebut, con_rebut,
                pro_close, con_close
            )
            answer_map[example_id] = {
                "intent": intent,
                "support_role": support_role,
                "oppose_role": oppose_role,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi":
            pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result = run_multi_agent(claim, evidence)
            answer_map[example_id] = {
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "intent_enhanced_single_sep":
            intent = infer_intent(claim)
            result = final_verdict(claim, evidence, intent)
            answer_map[example_id] = [result]

        elif args.mode == "intent_enhanced_multi_sep":
            evidence_transformed = f"### Pro-side Evidence:\n{evidence_pro_text}\n\n### Con-side Evidence:\n{evidence_con_text}"
            pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result = run_multi_agent(claim, evidence_transformed)
            answer_map[example_id] = {
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }
            
        elif args.mode == "multi_3p":
            (true_open, halftrue_open, false_open, 
             true_rebut, halftrue_rebut, false_rebut,
             true_close, halftrue_close, false_close, final_result) = run_multi_agent_3p(claim, evidence)
            answer_map[example_id] = {
                "true_opening": true_open,
                "halftrue_opening": halftrue_open,
                "false_opening": false_open,
                "true_rebuttal": true_rebut,
                "halftrue_rebuttal": halftrue_rebut,
                "false_rebuttal": false_rebut,
                "true_closing": true_close,
                "halftrue_closing": halftrue_close,
                "false_closing": false_close,
                "final_verdict": final_result
            }
            
        with open(output_file, "w") as f:
            json.dump(answer_map, f, indent=2)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(answer_map)} examples")

if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(">> Using CUDA device:", torch.cuda.current_device())
    print(">> Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    main()