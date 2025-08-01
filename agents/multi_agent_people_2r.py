
from model.loader import load_model
from prompts.templates_people import (
    get_system_prompt,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    judge_prompt_2r
)

# Global model info for batch processing
model_info = None

def set_model_info(info):
    """Set the global model info for batch processing"""
    global model_info
    model_info = info

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

def run_multi_agent_people_batch(claims, evidences, batch_size=8):
    """Run multi-agent people debate in batch for Qwen model (2 rounds: opening + rebuttal)"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Generate prompts
            pol_open_prompt = politician_opening_prompt(claim, evidence)
            sci_open_prompt = scientist_opening_prompt(claim, evidence)
            
            # Batch process opening statements
            pol_system = get_system_prompt("politician")
            sci_system = get_system_prompt("scientist")
            opening_results = run_model_batch(
                [pol_system, sci_system], 
                [pol_open_prompt, sci_open_prompt]
            )
            pol_open, sci_open = opening_results
            
            # Generate rebuttal prompts
            pol_rebut_prompt = politician_rebuttal_prompt(claim, evidence, sci_open)
            sci_rebut_prompt = scientist_rebuttal_prompt(claim, evidence, pol_open)
            
            # Batch process rebuttals
            rebuttal_results = run_model_batch(
                [pol_system, sci_system],
                [pol_rebut_prompt, sci_rebut_prompt]
            )
            pol_rebut, sci_rebut = rebuttal_results
            
            # Generate judge prompt (without closing statements)
            judge_prompt_text = judge_prompt_2r(
                claim, evidence,
                pol_open, sci_open,
                pol_rebut, sci_rebut
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt_text, max_tokens=400)
            
            batch_results.append({
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results

# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, opponent_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    prompt = judge_prompt_2r(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

# === Shared model runner ===
def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        if hasattr(first, 'chat') and hasattr(first.chat, 'completions'):
            # GPT model
            response = first.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Qwen model
            tokenizer, model = model_info
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
    else:
        raise ValueError("Invalid model_info format")