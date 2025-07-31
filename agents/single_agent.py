from model.loader import load_model
from prompts.templates import system_prompt_fact_checker, user_prompt_single_agent

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

def verify_claim(claim, evidence):
    """
    Verify the veracity of a given claim using retrieved evidence.
    Returns the model's classification and explanation.
    """
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    # Load prompt components
    system_prompt = system_prompt_fact_checker()
    user_prompt = user_prompt_single_agent(claim, evidence)

    # Generate response using the run_model function
    response = run_model(system_prompt, user_prompt)
    return response

def verify_claims_batch(claims, evidences):
    """
    Verify multiple claims in batch using retrieved evidence.
    Returns list of model's classifications and explanations.
    """
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    # Load prompt components
    system_prompt = system_prompt_fact_checker()
    system_prompts = [system_prompt] * len(claims)
    user_prompts = [user_prompt_single_agent(claim, evidence) for claim, evidence in zip(claims, evidences)]

    # Generate responses using batch processing
    responses = run_model_batch(system_prompts, user_prompts)
    return responses