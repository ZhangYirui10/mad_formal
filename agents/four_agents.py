from model.loader import load_model
from prompts.templates_four import (
    get_system_prompt,
    user_prompt_opening_pro1,
    user_prompt_opening_pro2,
    user_prompt_opening_con1,
    user_prompt_opening_con2,
    user_prompt_rebuttal_pro1,
    user_prompt_rebuttal_pro2,
    user_prompt_rebuttal_con1,
    user_prompt_rebuttal_con2,
    user_prompt_closing_pro1,
    user_prompt_closing_pro2,
    user_prompt_closing_con1,
    user_prompt_closing_con2,
    user_prompt_judge_4_agents
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

# === Individual Agent Functions ===
# === Pro-1 Agent (Factual Expert) ===
def opening_pro1(claim, evidence):
    prompt = user_prompt_opening_pro1(claim, evidence)
    return run_model(get_system_prompt("pro1"), prompt)

def rebuttal_pro1(claim, evidence, con1_argument, con2_argument):
    prompt = user_prompt_rebuttal_pro1(claim, evidence, con1_argument, con2_argument)
    return run_model(get_system_prompt("pro1"), prompt)

def closing_pro1(claim, evidence):
    prompt = user_prompt_closing_pro1(claim, evidence)
    return run_model(get_system_prompt("pro1"), prompt)

# === Pro-2 Agent (Reasoning Expert) ===
def opening_pro2(claim, evidence):
    prompt = user_prompt_opening_pro2(claim, evidence)
    return run_model(get_system_prompt("pro2"), prompt)

def rebuttal_pro2(claim, evidence, con1_argument, con2_argument):
    prompt = user_prompt_rebuttal_pro2(claim, evidence, con1_argument, con2_argument)
    return run_model(get_system_prompt("pro2"), prompt)

def closing_pro2(claim, evidence):
    prompt = user_prompt_closing_pro2(claim, evidence)
    return run_model(get_system_prompt("pro2"), prompt)

# === Con-1 Agent (Source Critic) ===
def opening_con1(claim, evidence):
    prompt = user_prompt_opening_con1(claim, evidence)
    return run_model(get_system_prompt("con1"), prompt)

def rebuttal_con1(claim, evidence, pro1_argument, pro2_argument):
    prompt = user_prompt_rebuttal_con1(claim, evidence, pro1_argument, pro2_argument)
    return run_model(get_system_prompt("con1"), prompt)

def closing_con1(claim, evidence):
    prompt = user_prompt_closing_con1(claim, evidence)
    return run_model(get_system_prompt("con1"), prompt)

# === Con-2 Agent (Reasoning Critic) ===
def opening_con2(claim, evidence):
    prompt = user_prompt_opening_con2(claim, evidence)
    return run_model(get_system_prompt("con2"), prompt)

def rebuttal_con2(claim, evidence, pro1_argument, pro2_argument):
    prompt = user_prompt_rebuttal_con2(claim, evidence, pro1_argument, pro2_argument)
    return run_model(get_system_prompt("con2"), prompt)

def closing_con2(claim, evidence):
    prompt = user_prompt_closing_con2(claim, evidence)
    return run_model(get_system_prompt("con2"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pro1_open, pro2_open, con1_open, con2_open, 
                       pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
                       pro1_close, pro2_close, con1_close, con2_close):
    prompt = user_prompt_judge_4_agents(
        claim, evidence,
        pro1_open, pro2_open, con1_open, con2_open,
        pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
        pro1_close, pro2_close, con1_close, con2_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)