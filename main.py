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
from agents.multi_agent_people import (
    opening_politician, rebuttal_politician, closing_politician,
    opening_scientist, rebuttal_scientist, closing_scientist,
    judge_final_verdict as judge_final_verdict_people
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

def run_multi_agent_people(claim, evidence):
    print("\n=== Running Multi-Agent People Debate (Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    final_result = judge_final_verdict_people(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    return pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close, final_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "multi_people"],
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
        evidence = example["evidence_full_text"]

        if args.mode == "single":
            result = run_single_agent(claim, evidence)
            answer_map[example_id] = [result]

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

        elif args.mode == "multi_people":
            pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close, final_result = run_multi_agent_people(claim, evidence)
            answer_map[example_id] = {
                "pol_opening": pol_open,
                "sci_opening": sci_open,
                "pol_rebuttal": pol_rebut,
                "sci_rebuttal": sci_rebut,
                "pol_closing": pol_close,
                "sci_closing": sci_close,
                "final_verdict": final_result
            }
            
        with open(output_file, "w") as f:
            json.dump(answer_map, f, indent=2)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(answer_map)} examples")

if __name__ == "__main__":
    main()