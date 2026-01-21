from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
import torch

MODEL = "teknium/OpenHermes-2.5-Mistral-7B"

dataset = load_dataset("json", data_files="data/train_global.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora,
    dataset_text_field="instruction",
    tokenizer=tokenizer,
    max_seq_length=1024
)

trainer.train()
trainer.save_model("models/rookus-global-lora")
