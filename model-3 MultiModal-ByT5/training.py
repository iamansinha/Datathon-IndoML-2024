import os
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MinMaxScaler
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_train = None
train_data = read_jsonl('../split data/train.json', nrows=nrows_train)

nrows_val = None
val_data = read_jsonl('../split data/validation.json', nrows=nrows_val)

#-----------------------

# create a directory to store any preprocessing files
os.makedirs("util_files", exist_ok=True)

# Get unique retailer names and map to IDs
unique_retailers = train_data['retailer'].unique()
num_retailers = len(unique_retailers)
retailer_to_id = {name: idx for idx, name in enumerate(unique_retailers)}

# Save the retailer-to-ID mapping for later use in inference
with open('./util_files/retailer_to_id.json', 'w') as f:
    json.dump(retailer_to_id, f)

# Apply the mapping to create 'retailer_id'
train_data['retailer_id'] = train_data['retailer'].map(retailer_to_id)
val_data['retailer_id'] = val_data['retailer'].map(retailer_to_id)


def preprocess_data(data, which_set):
    # Assuming retailer ID is already encoded in data
    data['input_text'] = data['description']
    data['target_text'] = data.apply(lambda row: f"supergroup: {row['supergroup']}, group: {row['group']}, module: {row['module']}, brand: {row['brand']}", axis=1)
    data['retailer_id'] = data['retailer_id'].astype(int)

    # Handle missing prices by filling NaNs with -1
    data['price'] = data['price'].fillna(-1)
    
    if which_set == 'train':
        # Normalize the price (using MinMaxScaler)
        price_scaler = MinMaxScaler()
        # Fit MinMaxScaler on valid prices only (i.e., prices that are not -1)
        valid_train_prices = data[data['price'] != -1][['price']]
        data.loc[data['price'] != -1, 'price'] = price_scaler.fit_transform(valid_train_prices)
        # Save the scaler for later use
        joblib.dump(price_scaler, './util_files/price_scaler.pkl')
    elif which_set in ['val', 'test']:
        price_scaler = joblib.load('./util_files/price_scaler.pkl')
        # Apply the scaler only to valid prices (i.e., prices not equal to -1)
        valid_prices = data[data['price'] != -1][['price']]
        data.loc[data['price'] != -1, 'price'] = price_scaler.transform(valid_prices)
    
    return data[['input_text', 'target_text', 'retailer_id', 'price']]

# Preprocess train and validation datasets
train_processed = preprocess_data(train_data, which_set='train')
val_processed = preprocess_data(val_data, which_set='val')

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_processed)
val_dataset = Dataset.from_pandas(val_processed)

# Creating Dataset Dictionary
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})


class ByT5Multimodal(T5ForConditionalGeneration):
    def __init__(self, config, num_retailers, embedding_dim=8):
        super(ByT5Multimodal, self).__init__(config)
        
        # Define retailer embedding layer
        self.retailer_embedding = nn.Embedding(num_retailers, embedding_dim)
        
        # Define price projection layer (projecting scalar price to embedding dimension)
        self.price_projection = nn.Linear(1, embedding_dim)
        
        # Fusion Layer to combine ByT5 hidden states and retailer/price embeddings
        self.fusion_layer = nn.Linear(config.d_model + 2 * embedding_dim, config.d_model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        retailer_ids=None,
        price=None,
        **kwargs
    ):
        # Get ByT5 encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get text embedding from encoder output
        text_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Get retailer embedding
        retailer_emb = self.retailer_embedding(retailer_ids)
        
        # Project price to an embedding
        price = price.unsqueeze(1)
        price_emb = self.price_projection(price)
        
        # Concatenate and fuse embeddings
        combined_emb = torch.cat([text_embedding, retailer_emb, price_emb], dim=1)
        fused_emb = self.fusion_layer(combined_emb)
        
        # Assign the fused embedding to the encoder [CLS] token
        encoder_hidden_states = encoder_outputs.last_hidden_state
        enhanced_hidden_states = encoder_hidden_states.clone()
        enhanced_hidden_states[:, 0, :] = fused_emb  # Only modify the [CLS] token
        
        # Prepare decoder inputs
        if labels is not None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        else:
            decoder_input_ids = None

        # Run the decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enhanced_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "encoder_last_hidden_state": enhanced_hidden_states,
        }
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'google/byt5-small'                   # 300M size

# Instantiate the custom ByT5Multimodal model
model = ByT5Multimodal.from_pretrained(model_id, num_retailers=num_retailers, embedding_dim=8).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    retailer_ids = examples['retailer_id']
    prices = examples['price']

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=256, padding="max_length", truncation=True, return_tensors="pt")

    # Set labels
    model_inputs['labels'] = labels['input_ids']

    # Add retailer ID and price to the input features
    model_inputs['retailer_ids'] = torch.tensor(retailer_ids)
    model_inputs['price'] = torch.tensor(prices, dtype=torch.float)

    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, num_proc=32)

# Define the output and logging directories
new_model_name = 'multi-modal-byt5-small'
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

def save_model_contiguous(model, model_path):
    state_dict = model.state_dict()
    contiguous_state_dict = {k: v.contiguous() if not v.is_contiguous() else v for k, v in state_dict.items()}
    model.save_pretrained(model_path, state_dict=contiguous_state_dict)

model_path = training_args.output_dir + '-final'
save_model_contiguous(model, model_path)
tokenizer.save_pretrained(model_path)
