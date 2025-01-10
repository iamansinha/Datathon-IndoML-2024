import pandas as pd
from huggingface_hub import login
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

login(token='YOUR_TOKEN')  # Replace 'YOUR_TOKEN' with your actual huggingface token

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_train = None
data_path = "./train_data_features_with_description_inferences.json"
train_data = read_jsonl(data_path, nrows=nrows_train)

#-------------------------------------

model_id = "meta-llama/Llama-3.1-8B-Instruct"

max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    device_map='cuda',
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def create_messages(examples):
    """
    Generates chat-based format message prompts based on a batch of examples.
    Each example corresponds to one row of the dataset.
    """
    messages = []
    # Iterate over the batch of examples (each field in examples is a list)
    for i in range(len(examples['description'])):
        # Create a message for each example in the batch
        message = [
            {"role": "system", "content": "You are a knowledgeable chatbot who generates product name."},
            {"role": "user", "content": f"""
Product Description: {examples['description'][i]}
Price: ${examples['price'][i]}

Using the provided details, guess the most accurate and complete product name as per your knowledge of products of all the brands. Ensure that any concatenated words in the description are separated appropriately. The output should consist of one full product name only, without any additional explanation or formatting."""},
            {"role": "assistant", "content": f"{examples['infer_desc'][i]}"}
        ]
        messages.append(message)
    
    return messages


def apply_template(examples):
    messages = create_messages(examples)
    texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": texts}

def load_custom_dataset(file_path, split='train'):
    return load_dataset('json', data_files={split: file_path}, split=split)

dataset = load_custom_dataset(data_path, split="train")
dataset = dataset.map(apply_template, batched=True)

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        save_strategy="epoch",
        output_dir="output",
        seed=55,
    ),
)

trainer.train()

model.save_pretrained_merged("Llama-3.1-8B-Instruct-for-Datathon", tokenizer, save_method="merged_16bit")