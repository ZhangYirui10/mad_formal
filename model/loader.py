from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

def load_model(model_path=None):
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "llama3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model