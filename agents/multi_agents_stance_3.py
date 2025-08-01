from model.loader import load_model
from prompts.templates_stance_3 import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_opening_flexible,
    user_prompt_rebuttal_flexible,
    user_prompt_closing_flexible,
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

# === Pro Agent ===
def opening_pro(claim, evidence):
    prompt = user_prompt_opening_pro(claim, evidence)
    return run_model(get_system_prompt("pro"), prompt)

def rebuttal_pro(claim, evidence, con_argument):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_argument)
    return run_model(get_system_prompt("pro"), prompt)

def closing_pro(claim, evidence):
    prompt = user_prompt_closing_pro(claim, evidence)
    return run_model(get_system_prompt("pro"), prompt)

# === Con Agent ===
def opening_con(claim, evidence):
    prompt = user_prompt_opening_con(claim, evidence)
    return run_model(get_system_prompt("con"), prompt)

def rebuttal_con(claim, evidence, pro_argument):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_argument)
    return run_model(get_system_prompt("con"), prompt)

def closing_con(claim, evidence):
    prompt = user_prompt_closing_con(claim, evidence)
    return run_model(get_system_prompt("con"), prompt)

# === Flexible Agent ===
def opening_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_opening_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

def rebuttal_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_rebuttal_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

def closing_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_closing_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, flexible_open, pro_open, con_open, flexible_rebut, pro_rebut, con_rebut, flexible_close, pro_close, con_close):
    prompt = judge_prompt_three_agents(
        claim, evidence,
        flexible_open, pro_open, con_open,
        flexible_rebut, pro_rebut, con_rebut,
        flexible_close, pro_close, con_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

def run_multi_agent_stance_3_batch(claims, evidences, batch_size=8):
    """Run multi-agent stance 3 debate in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Step 1: Generate opening statements for pro and con first
            pro_open_prompt = user_prompt_opening_pro(claim, evidence)
            con_open_prompt = user_prompt_opening_con(claim, evidence)
            
            # Batch process pro and con opening statements
            pro_system = get_system_prompt("pro")
            con_system = get_system_prompt("con")
            opening_results = run_model_batch(
                [pro_system, con_system], 
                [pro_open_prompt, con_open_prompt]
            )
            pro_open, con_open = opening_results
            
            # Step 2: Generate flexible opening statement (needs pro and con arguments)
            flex_open_prompt = user_prompt_opening_flexible(claim, evidence, pro_open, con_open)
            flex_system = get_system_prompt("flexible")
            flex_open = run_model(flex_system, flex_open_prompt)
            
            # Step 3: Generate rebuttal prompts
            pro_rebut_prompt = user_prompt_rebuttal_pro(claim, evidence, con_open)
            con_rebut_prompt = user_prompt_rebuttal_con(claim, evidence, pro_open)
            
            # Batch process pro and con rebuttals
            rebuttal_results = run_model_batch(
                [pro_system, con_system],
                [pro_rebut_prompt, con_rebut_prompt]
            )
            pro_rebut, con_rebut = rebuttal_results
            
            # Step 4: Generate flexible rebuttal (needs pro and con arguments)
            flex_rebut_prompt = user_prompt_rebuttal_flexible(claim, evidence, pro_rebut, con_rebut)
            flex_rebut = run_model(flex_system, flex_rebut_prompt)
            
            # Step 5: Generate closing prompts
            pro_close_prompt = user_prompt_closing_pro(claim, evidence)
            con_close_prompt = user_prompt_closing_con(claim, evidence)
            
            # Batch process pro and con closings
            closing_results = run_model_batch(
                [pro_system, con_system],
                [pro_close_prompt, con_close_prompt]
            )
            pro_close, con_close = closing_results
            
            # Step 6: Generate flexible closing (needs pro and con arguments)
            flex_close_prompt = user_prompt_closing_flexible(claim, evidence, pro_close, con_close)
            flex_close = run_model(flex_system, flex_close_prompt)
            
            # Step 7: Generate judge prompt
            judge_prompt_text = judge_prompt_three_agents(
                claim, evidence,
                flex_open, pro_open, con_open,
                flex_rebut, pro_rebut, con_rebut,
                flex_close, pro_close, con_close
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt_text, max_tokens=400)
            
            batch_results.append({
                "flexible_opening": flex_open,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "flexible_rebuttal": flex_rebut,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "flexible_closing": flex_close,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results