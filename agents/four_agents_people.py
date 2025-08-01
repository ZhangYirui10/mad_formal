from model.loader import load_model
from prompts.templates_four_people import (
    get_system_prompt,
    user_prompt_opening_politician,
    user_prompt_rebuttal_politician,
    user_prompt_closing_politician,
    user_prompt_opening_scientist,
    user_prompt_rebuttal_scientist,
    user_prompt_closing_scientist,
    user_prompt_opening_journalist,
    user_prompt_rebuttal_journalist,
    user_prompt_closing_journalist,
    user_prompt_opening_domain_scientist,
    user_prompt_rebuttal_domain_scientist,
    user_prompt_closing_domain_scientist,
    user_prompt_judge_4_agents,
    user_prompt_domain_inference
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

# === Domain Specialist Inference ===
def infer_domain_specialist(claim):
    prompt = user_prompt_domain_inference(claim)
    domain_output = run_model(get_system_prompt("fact_checker"), prompt, max_tokens=100)
    
    # Parse domain specialist from output
    domain_specialist = "Domain Expert"
    for line in domain_output.splitlines():
        if line.startswith("DOMAIN:"):
            domain_specialist = line.split(":", 1)[1].strip()
            break
    
    return domain_specialist

# === Individual Agent Functions ===
# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = user_prompt_opening_politician(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, scientist_argument, journalist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_politician(claim, evidence, scientist_argument, journalist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    prompt = user_prompt_closing_politician(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = user_prompt_opening_scientist(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, politician_argument, journalist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_scientist(claim, evidence, politician_argument, journalist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = user_prompt_closing_scientist(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Journalist Agent ===
def opening_journalist(claim, evidence):
    prompt = user_prompt_opening_journalist(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

def rebuttal_journalist(claim, evidence, politician_argument, scientist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_journalist(claim, evidence, politician_argument, scientist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("journalist"), prompt)

def closing_journalist(claim, evidence):
    prompt = user_prompt_closing_journalist(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

# === Domain Scientist Agent ===
def opening_domain_scientist(claim, evidence, domain_specialist: str = None):
    prompt = user_prompt_opening_domain_scientist(claim, evidence, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

def rebuttal_domain_scientist(claim, evidence, politician_argument, scientist_argument, journalist_argument, domain_specialist: str = None):
    prompt = user_prompt_rebuttal_domain_scientist(claim, evidence, politician_argument, scientist_argument, journalist_argument, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

def closing_domain_scientist(claim, evidence, domain_specialist: str = None):
    prompt = user_prompt_closing_domain_scientist(claim, evidence, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, politician_open, scientist_open, journalist_open, domain_scientist_open, 
                       politician_rebut, scientist_rebut, journalist_rebut, domain_scientist_rebut,
                       politician_close, scientist_close, journalist_close, domain_scientist_close):
    prompt = user_prompt_judge_4_agents(
        claim, evidence, politician_open, scientist_open, journalist_open, domain_scientist_open,
        politician_rebut, scientist_rebut, journalist_rebut, domain_scientist_rebut,
        politician_close, scientist_close, journalist_close, domain_scientist_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)