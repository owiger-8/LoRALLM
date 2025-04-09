# 0. Install dependencies first (run once in terminal)
# pip install -U transformers peft accelerate bitsandbytes trl datasets

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Convert dataset format with proper encoding
try:
    with open("data.json", encoding="utf-8") as f:
        original_data = json.load(f)
except UnicodeDecodeError:
    with open("data.json", encoding="utf-8-sig") as f:
        original_data = json.load(f)

formatted_data = []
for item in original_data:
    formatted_data.append({
        "messages": [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]
    })

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")

# 2. Load model with 4-bit quantization
model_id = "cognitivecomputations/TinyDolphin-2.8.2-1.1b-laser"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Set custom chat template
tokenizer.chat_template = """{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{% endfor %}"""
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Load formatted dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# 5. Configure training parameters
training_config = SFTConfig(
    max_seq_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="tinydolphin-finetuned",
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    report_to="none"
)

# 6. Formatting function
def format_instruction(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

# 7. Set up trainer with corrected parameter name
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=format_instruction,
    processing_class=tokenizer,  # Changed from tokenizer to processing_class
    args=training_config
)

# 8. Start training
trainer.train()

# 9. Save adapter and tokenizer
trainer.model.save_pretrained("tinydolphin-finetuned")
tokenizer.save_pretrained("tinydolphin-finetuned")
