from model.loader import load_model
from prompts.templates import (
    get_system_prompt,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con
)
# Load model once for all agents
tokenizer, model = load_model()
def infer_intent(claim):
    prompt = user_prompt_intent_inference(claim)
    return run_model(get_system_prompt("fact_checker"), prompt, max_tokens=150)

def reformulate_claim_pro(claim, intent):
    prompt = user_prompt_reformulate_pro(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=100)

def reformulate_claim_con(claim, intent):
    prompt = user_prompt_reformulate_con(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=100)

# === Shared model runner ===
def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
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


def intent_enhanced_reformulation(claim: str):

    # Step 1: Infer Intent
    intent = infer_intent(claim)
    # print(f"[Inferred Intent]: {intent}")

    # Step 2: Reformulate Pro Version
    reformulated_pro = reformulate_claim_pro(claim, intent)
    # print(f"[Pro Reformulated Claim]: {reformulated_pro}")

    # Step 3: Reformulate Con Version
    reformulated_con = reformulate_claim_con(claim, intent)
    # print(f"[Con Reformulated Claim]: {reformulated_con}")

    return {
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con
    }