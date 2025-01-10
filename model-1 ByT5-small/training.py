import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_train = None
train_data = read_jsonl('../split data/train.json', nrows=nrows_train)

nrows_val = None
val_data = read_jsonl('../split data/validation.json', nrows=nrows_val)

def preprocess_data(data):
    data['input_text'] = data.apply(lambda row: f"{row['description']}", axis=1)
    data['target_text'] = data.apply(lambda row: f"supergroup: {row['supergroup']}, group: {row['group']}, module: {row['module']}, brand: {row['brand']}", axis=1)
    
    return data[['input_text', 'target_text']]

train_processed = preprocess_data(train_data)
val_processed = preprocess_data(val_data)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_processed)
val_dataset = Dataset.from_pandas(val_processed)

# Creating Dataset Dictionary
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

model_id = 'google/byt5-small'                   # 300M size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']

    model_inputs = tokenizer(inputs, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=256, padding="max_length", truncation=True, return_tensors="pt")

    # Set labels
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, num_proc=32)

#--------------------------------------------

# Define the output and logging directories
new_model_name = 'byt5-small-ft-with-only-desc'
output_dir = './' + new_model_name
logging_dir = f'./logs'

# Ensure the logging directory exists
os.makedirs(logging_dir, exist_ok=True)

# Define the log file path
log_file = f'{logging_dir}/{new_model_name}-training.log'

# Ensure the log file is overwritten at the start
open(log_file, 'w').close()

# Custom callback to log training metrics and arguments
class CustomCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_train_begin(self, args, state, control, **kwargs):
        # Log training arguments at the start (overwrite existing content)
        with open(self.log_file, 'w') as f:  # 'w' ensures overwriting
            f.write("Training Arguments:\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file, 'a') as f:  # Append logs during training
                f.write(f"Step: {state.global_step}\n")
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")


# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=500,
    push_to_hub=False,
    report_to="none",
    save_safetensors=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CustomCallback(log_file)]
)

# Train the model
trainer.train()


# Function to save model with contiguous tensors
def save_model_contiguous(model, model_path):
    # Get the model state_dict
    state_dict = model.state_dict()

    # Make all tensors contiguous
    contiguous_state_dict = {k: v.contiguous() if not v.is_contiguous() else v for k, v in state_dict.items()}

    # Save the model with the updated state_dict
    model.save_pretrained(model_path, state_dict=contiguous_state_dict)

# Define the final model save path
model_path = training_args.output_dir + '-final'

# Save the model with contiguous tensors
save_model_contiguous(model, model_path)

# Save the tokenizer as usual (no changes needed here)
tokenizer.save_pretrained(model_path)
