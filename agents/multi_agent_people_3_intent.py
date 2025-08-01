
from model.loader import load_model
from prompts.templates_people_3 import (
    get_system_prompt,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_closing_prompt,
    journalist_opening_prompt,
    journalist_rebuttal_prompt,
    journalist_closing_prompt,
    judge_prompt_three_agents
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
    
# === Intent Inference ===
def infer_intent(claim):
    """Infer the intended message of a claim"""
    prompt = user_prompt_intent_inference(claim)
    return run_model(get_system_prompt("fact_checker"), prompt, max_tokens=100)

# === Claim Reformulation ===
def reformulate_claim_pro(claim, intent):
    """Reformulate claim from pro perspective"""
    prompt = user_prompt_reformulate_pro(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

def reformulate_claim_con(claim, intent):
    """Reformulate claim from con perspective"""
    prompt = user_prompt_reformulate_con(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

# === Journalist Agent ===
def opening_journalist(claim, evidence):
    prompt = journalist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

def rebuttal_journalist(claim, evidence, politician_argument, scientist_argument):
    prompt = journalist_rebuttal_prompt(claim, evidence, politician_argument, scientist_argument)
    return run_model(get_system_prompt("journalist"), prompt)

def closing_journalist(claim, evidence, politician_rebuttal, scientist_rebuttal):
    prompt = journalist_closing_prompt(claim, evidence, politician_rebuttal, scientist_rebuttal)
    return run_model(get_system_prompt("journalist"), prompt)

# === Politician Agent ===
def opening_politician(claim, evidence, journalist_argument):
    prompt = politician_opening_prompt(claim, evidence, journalist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, scientist_argument, journalist_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, scientist_argument, journalist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence, journalist_rebuttal):
    prompt = politician_closing_prompt(claim, evidence, journalist_rebuttal)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence, journalist_argument):
    prompt = scientist_opening_prompt(claim, evidence, journalist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, politician_argument, journalist_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, politician_argument, journalist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence, journalist_rebuttal):
    prompt = scientist_closing_prompt(claim, evidence, journalist_rebuttal)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close):
    prompt = judge_prompt_three_agents(
        claim, evidence,
        jour_open, pol_open, sci_open,
        jour_rebut, pol_rebut, sci_rebut,
        jour_close, pol_close, sci_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)