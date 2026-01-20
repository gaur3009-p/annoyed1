from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.7,
        do_sample=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
