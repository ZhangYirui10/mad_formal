
from model.loader import load_model
from prompts.templates_people import (
    get_system_prompt,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_closing_prompt,
    judge_prompt
)

# Load model once for all agents
tokenizer, model = load_model()

# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    prompt = politician_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, opponent_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    prompt = judge_prompt(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

# === Shared model runner ===
def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
    full_prompt = f"<|begin_of_text|><|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()