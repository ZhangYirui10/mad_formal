from model.loader import load_model
from prompts.templates import system_prompt_fact_checker, user_prompt_single_agent

# Load model and tokenizer once
tokenizer, model = load_model()


def verify_claim(claim, evidence):
    """
    Verify the veracity of a given claim using retrieved evidence.
    Returns the model's classification and explanation.
    """
    # Load prompt components
    system_prompt = system_prompt_fact_checker()
    user_prompt = user_prompt_single_agent(claim, evidence)

    # Assemble final input prompt with special tokens
    full_prompt = f"<|begin_of_text|><|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Run model inference
    outputs = model.generate(
        **inputs,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and return the result (trim to only assistant's answer)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    return response