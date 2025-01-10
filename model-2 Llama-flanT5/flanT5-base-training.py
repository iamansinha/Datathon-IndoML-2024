import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

train_features = read_jsonl('./train_data_features_with_description_inferences.json')
train_labels = read_jsonl('../datathon_phase_2_data/training_data/train.labels')
train_all_data = pd.merge(train_features, train_labels, on='indoml_id', how='inner')

columns = ['supergroup', 'group', 'module', 'brand']

def sample_n_percent(group):
    n = 0.1     # 10% split
    return group.sample(frac=n, random_state=42)

# Group by the specified columns and sample 10% from each group
val_data = train_all_data.groupby(columns, group_keys=False).apply(sample_n_percent)

# Remove the sampled rows from the original train_data
train_data = train_all_data.drop(val_data.index)

# Sort both DataFrames by 'indoml_id' and reset index
train_data = train_data.sort_values(by='indoml_id').reset_index(drop=True)
val_data = val_data.sort_values(by='indoml_id').reset_index(drop=True)

#--------------------

def preprocess_data(data):
    data['input_text'] = data.apply(lambda row: f"{row['infer_desc']}", axis=1)
    data['target_text'] = data.apply(lambda row: f"supergroup: {row['supergroup']}, group: {row['group']}, module: {row['module']}, brand: {row['brand']}", axis=1)
    
    return data[['input_text', 'target_text']]

train_processed = preprocess_data(train_data)
val_processed = preprocess_data(val_data)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_processed)
val_dataset = Dataset.from_pandas(val_processed)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'google/flan-t5-base'    # 248M params
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']

    model_inputs = tokenizer(inputs, max_length=150, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=256, padding="max_length", truncation=True, return_tensors="pt")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)


# Define the output and logging directories
new_model_name = 'flan-t5-base-ft'
output_dir = './' + new_model_name
logging_dir = f'./logs'

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
                    # print(f"{key}: {value}")
                f.write("\n")
                # print()


# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=5000,
    push_to_hub=False,
    report_to="none",
    save_safetensors=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CustomCallback(log_file)]
)

# Train the model
trainer.train()


def save_model_contiguous(model, model_path):
    state_dict = model.state_dict()
    contiguous_state_dict = {k: v.contiguous() if not v.is_contiguous() else v for k, v in state_dict.items()}
    model.save_pretrained(model_path, state_dict=contiguous_state_dict)

model_path = training_args.output_dir + '-final'
save_model_contiguous(model, model_path)
tokenizer.save_pretrained(model_path)
