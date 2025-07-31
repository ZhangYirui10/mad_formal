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

def run_model_batch(system_prompts: list, user_prompts: list, max_tokens: int = 300):
    """Run model inference in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        if hasattr(first, 'chat') and hasattr(first.chat, 'completions'):
            # GPT model - process individually
            results = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                result = run_model(sys_prompt, usr_prompt, max_tokens)
                results.append(result)
            return results
        else:
            # Qwen model - batch processing
            tokenizer, model = model_info
            full_prompts = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                full_prompt = f"<|begin_of_text|><|system|>\n{sys_prompt}\n<|user|>\n{usr_prompt}<|assistant|>\n"
                full_prompts.append(full_prompt)
            
            # Tokenize all prompts
            inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Generate in batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode all outputs
            responses = []
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                # Extract assistant response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                responses.append(response)
            
            return responses
    
    else:
        raise ValueError("Invalid model_info format")

def run_four_agents_batch(claims, evidences, batch_size=8):
    """Run 4-agent debate in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Generate opening prompts for all 4 agents
            pro1_open_prompt = user_prompt_opening_pro1(claim, evidence)
            pro2_open_prompt = user_prompt_opening_pro2(claim, evidence)
            con1_open_prompt = user_prompt_opening_con1(claim, evidence)
            con2_open_prompt = user_prompt_opening_con2(claim, evidence)
            
            # Batch process opening statements
            pro1_system = get_system_prompt("pro1")
            pro2_system = get_system_prompt("pro2")
            con1_system = get_system_prompt("con1")
            con2_system = get_system_prompt("con2")
            
            opening_results = run_model_batch(
                [pro1_system, pro2_system, con1_system, con2_system], 
                [pro1_open_prompt, pro2_open_prompt, con1_open_prompt, con2_open_prompt]
            )
            pro1_open, pro2_open, con1_open, con2_open = opening_results
            
            # Generate rebuttal prompts
            pro1_rebut_prompt = user_prompt_rebuttal_pro1(claim, evidence, con1_open, con2_open)
            pro2_rebut_prompt = user_prompt_rebuttal_pro2(claim, evidence, con1_open, con2_open)
            con1_rebut_prompt = user_prompt_rebuttal_con1(claim, evidence, pro1_open, pro2_open)
            con2_rebut_prompt = user_prompt_rebuttal_con2(claim, evidence, pro1_open, pro2_open)
            
            # Batch process rebuttals
            rebuttal_results = run_model_batch(
                [pro1_system, pro2_system, con1_system, con2_system],
                [pro1_rebut_prompt, pro2_rebut_prompt, con1_rebut_prompt, con2_rebut_prompt]
            )
            pro1_rebut, pro2_rebut, con1_rebut, con2_rebut = rebuttal_results
            
            # Generate closing prompts
            pro1_close_prompt = user_prompt_closing_pro1(claim, evidence)
            pro2_close_prompt = user_prompt_closing_pro2(claim, evidence)
            con1_close_prompt = user_prompt_closing_con1(claim, evidence)
            con2_close_prompt = user_prompt_closing_con2(claim, evidence)
            
            # Batch process closings
            closing_results = run_model_batch(
                [pro1_system, pro2_system, con1_system, con2_system],
                [pro1_close_prompt, pro2_close_prompt, con1_close_prompt, con2_close_prompt]
            )
            pro1_close, pro2_close, con1_close, con2_close = closing_results
            
            # Generate judge prompt
            judge_prompt = user_prompt_judge_4_agents(
                claim, evidence,
                pro1_open, pro2_open, con1_open, con2_open,
                pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
                pro1_close, pro2_close, con1_close, con2_close
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt, max_tokens=400)
            
            batch_results.append({
                "pro1_opening": pro1_open,
                "pro2_opening": pro2_open,
                "con1_opening": con1_open,
                "con2_opening": con2_open,
                "pro1_rebuttal": pro1_rebut,
                "pro2_rebuttal": pro2_rebut,
                "con1_rebuttal": con1_rebut,
                "con2_rebuttal": con2_rebut,
                "pro1_closing": pro1_close,
                "pro2_closing": pro2_close,
                "con1_closing": con1_close,
                "con2_closing": con2_close,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results

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