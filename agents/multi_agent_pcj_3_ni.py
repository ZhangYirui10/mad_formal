from model.loader import load_model
from prompts.templates_pcj_3 import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_pro,
    user_prompt_rebuttal_con,
    user_prompt_closing_pro,
    user_prompt_closing_con,
    user_prompt_judge_full,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con,
    user_prompt_opening_journalist,
    user_prompt_rebuttal_journalist,
    user_prompt_closing_journalist
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

# === Pro Agent ===
def opening_pro(claim, evidence):
    """Pro agent opening statement"""
    prompt = user_prompt_opening_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_pro(claim, evidence, con_argument):
    """Pro agent rebuttal"""
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_argument)
    return run_model(get_system_prompt("debater"), prompt)

def closing_pro(claim, evidence):
    """Pro agent closing statement"""
    prompt = user_prompt_closing_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

# === Con Agent ===
def opening_con(claim, evidence):
    """Con agent opening statement"""
    prompt = user_prompt_opening_con(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_con(claim, evidence, pro_argument):
    """Con agent rebuttal"""
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_argument)
    return run_model(get_system_prompt("debater"), prompt)

def closing_con(claim, evidence):
    """Con agent closing statement"""
    prompt = user_prompt_closing_con(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

# === Journalist Agent ===
def opening_journalist(claim, evidence):
    """Journalist opening analysis"""
    prompt = user_prompt_opening_journalist(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

def rebuttal_journalist(claim, evidence, pro_argument, con_argument):
    """Journalist rebuttal analysis"""
    prompt = user_prompt_rebuttal_journalist(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("journalist"), prompt)

def closing_journalist(claim, evidence, pro_rebuttal, con_rebuttal):
    """Journalist closing analysis"""
    prompt = user_prompt_closing_journalist(claim, evidence, pro_rebuttal, con_rebuttal)
    return run_model(get_system_prompt("journalist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pro_open, con_open, jour_open, pro_rebut, con_rebut, jour_rebut, pro_close, con_close, jour_close):
    """Judge final verdict"""
    prompt = user_prompt_judge_full(
        claim, evidence,
        pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

# === Main Debate Function ===
def run_full_debate(claim, evidence):
    """Run the complete debate with all agents"""
    
    # Step 3: Opening statements
    jour_open = opening_journalist(claim, evidence)
    pro_open = opening_pro(claim, evidence)
    con_open = opening_con(claim, evidence)
    
    # Step 4: Rebuttals
    jour_rebut = rebuttal_journalist(claim, evidence, pro_open, con_open)
    pro_rebut = rebuttal_pro(claim, evidence, con_open)
    con_rebut = rebuttal_con(claim, evidence, pro_open)
    
    # Step 5: Closing statements
    jour_close = closing_journalist(claim, evidence, pro_rebut, con_rebut)
    pro_close = closing_pro(claim, evidence)
    con_close = closing_con(claim, evidence)
    
    # Step 6: Judge verdict
    verdict = judge_final_verdict(
        claim, evidence,
        pro_open, con_open, jour_open,
        pro_rebut, con_rebut, jour_rebut,
        pro_close, con_close, jour_close
    )
    
    return {
        "journalist_opening": jour_open,
        "pro_opening": pro_open,
        "con_opening": con_open,
        "journalist_rebuttal": jour_rebut,
        "pro_rebuttal": pro_rebut,
        "con_rebuttal": con_rebut,
        "journalist_closing": jour_close,
        "pro_closing": pro_close,
        "con_closing": con_close,
        "verdict": verdict
    }
