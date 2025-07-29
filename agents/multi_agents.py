from model.loader import load_model
from prompts.templates import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_judge_full
)

# Load model once for all agents
tokenizer, model = load_model()
    
# === Pro Agent ===
def opening_pro(claim, evidence):
    prompt = user_prompt_opening_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_pro(claim, evidence, con_opening_statement):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_opening_statement)
    return run_model(get_system_prompt("debater"), prompt)

def closing_pro(claim, evidence):
    prompt = user_prompt_closing_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)


# === Con Agent ===
def opening_con(claim, evidence):
    prompt = user_prompt_opening_con(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_con(claim, evidence, pro_opening_statement):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_opening_statement)
    return run_model(get_system_prompt("debater"), prompt)

def closing_con(claim, evidence):
    prompt = user_prompt_closing_con(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)


# === Judge Agent ===
def judge_final_verdict(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    prompt = user_prompt_judge_full(
        claim, evidence,
        pro_open, con_open,
        pro_rebut, con_rebut,
        pro_close, con_close
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
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()