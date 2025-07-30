from model.loader import load_model
from prompts.templates_role import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_judge_full,
    user_prompt_intent_inference,
    user_prompt_role_inference
)

# Global model info - will be set by main.py
model_info = None

def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info

def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
    """Run model inference based on model type"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        if hasattr(first, 'chat') and hasattr(first.chat, 'completions'):
            client, model_name = model_info
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        else:
            tokenizer, model = model_info
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
    
    else:
        raise ValueError("Invalid model_info format")

# === Step 1: Infer intent and roles ===
def infer_intent_and_roles(claim):
    intent_prompt = user_prompt_intent_inference(claim)
    intent = run_model(get_system_prompt("fact_checker"), intent_prompt)

    role_prompt = user_prompt_role_inference(intent)
    roles_output = run_model(get_system_prompt("fact_checker"), role_prompt)

    support_role = "Pro"
    oppose_role = "Con"
    for line in roles_output.splitlines():
        if line.startswith("SUPPORTING_ROLE:"):
            support_role = line.split(":", 1)[1].strip()
        elif line.startswith("OPPOSING_ROLE:"):
            oppose_role = line.split(":", 1)[1].strip()
    return intent, support_role, oppose_role

# === Pro Agent ===
def opening_pro(claim, evidence, role):
    prompt = user_prompt_opening_pro(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_pro(claim, evidence, con_opening_statement, role):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_opening_statement, role)
    return run_model(get_system_prompt("debater"), prompt)

def closing_pro(claim, evidence, role):
    prompt = user_prompt_closing_pro(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

# === Con Agent ===
def opening_con(claim, evidence, role):
    prompt = user_prompt_opening_con(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_con(claim, evidence, pro_opening_statement, role):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_opening_statement, role)
    return run_model(get_system_prompt("debater"), prompt)

def closing_con(claim, evidence, role):
    prompt = user_prompt_closing_con(claim, evidence, role)
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