import re
import re
import json
import joblib
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_test = None
test_data = read_jsonl('../phase_2_input_data/final_test_data/final_test_data.features', nrows=nrows_test)

# Load the retailer-to-ID mapping and MinMaxScaler
with open('./util_files/retailer_to_id.json', 'r') as f:
    retailer_to_id = json.load(f)
    
# Apply the mapping to create 'retailer_id'
test_data['retailer_id'] = test_data['retailer'].map(retailer_to_id)


def preprocess_data(data, which_set):
    # Assuming retailer ID is already encoded in data
    data['input_text'] = data['description']

    # Handle missing prices by filling NaNs with -1
    data['retailer_id'] = data['retailer_id'].fillna(0)
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
    
    return data[['indoml_id', 'input_text', 'retailer_id', 'price']]


test_processed = preprocess_data(test_data, which_set='test')
test_dataset = Dataset.from_pandas(test_processed)


class ByT5Multimodal(T5ForConditionalGeneration):
    def __init__(self, config, num_retailers, embedding_dim=8):
        super(ByT5Multimodal, self).__init__(config)
        
        # Define retailer embedding layer
        self.retailer_embedding = nn.Embedding(num_retailers, embedding_dim)
        
        # Define price projection layer (projecting scalar price to embedding dimension)
        self.price_projection = nn.Linear(1, embedding_dim)
        
        # Fusion Layer to combine ByT5 hidden states and retailer/price embeddings
        self.fusion_layer = nn.Linear(config.d_model + 2 * embedding_dim, config.d_model)

    def _get_encoder_outputs(
        self,
        input_ids=None,
        attention_mask=None,
        retailer_ids=None,
        price=None,
        encoder_outputs=None,
        **kwargs
    ):
        """Helper method to get encoder outputs with multimodal fusion"""
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # If retailer_ids and price are not provided (during generation),
        # return the original encoder outputs
        if retailer_ids is None or price is None:
            return encoder_outputs

        # Get text embedding from encoder output
        text_embedding = encoder_hidden_states[:, 0, :]
        
        # Get retailer embedding
        retailer_emb = self.retailer_embedding(retailer_ids)
        
        # Project price to an embedding
        price = price.unsqueeze(1)
        price_emb = self.price_projection(price)
        
        # Concatenate and fuse embeddings
        combined_emb = torch.cat([text_embedding, retailer_emb, price_emb], dim=1)
        fused_emb = self.fusion_layer(combined_emb)
        
        # Create enhanced hidden states with fused embedding
        enhanced_hidden_states = encoder_hidden_states.clone()
        enhanced_hidden_states[:, 0, :] = fused_emb  # Only modify the [CLS] token
        
        # Return modified encoder outputs
        encoder_outputs.last_hidden_state = enhanced_hidden_states
        return encoder_outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        retailer_ids=None,
        price=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle encoder outputs
        encoder_outputs = self._get_encoder_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            retailer_ids=retailer_ids,
            price=price,
            encoder_outputs=encoder_outputs
        )

        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        # Run the decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'multi-modal-byt5-small-final'
model = ByT5Multimodal.from_pretrained(model_id, num_retailers=63).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


def tokenize_batch(batch_data):
    """Tokenize the input text and prepare model inputs."""
    indoml_ids = batch_data['indoml_id']
    input_texts = batch_data['input_text']
    
    # Convert inputs to appropriate tensor types
    retailer_ids = torch.tensor(batch_data['retailer_id'], device=device)
    prices = torch.tensor(batch_data['price'], dtype=torch.float32, device=device)
    
    # Tokenize inputs
    encoded = tokenizer.batch_encode_plus(
        input_texts,
        return_tensors="pt",
        padding=True
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    return indoml_ids, {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'retailer_ids': retailer_ids,
        'price': prices
    }

def generate_text(tokenized_inputs_with_ids):
    """Generate text using the model."""
    indoml_ids, tokenized_inputs = tokenized_inputs_with_ids
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_inputs,
            max_length=256,  # Adjust as needed
        )
    return indoml_ids, outputs


def decode_outputs(outputs_with_ids):
    """Decode the generated outputs."""
    indoml_ids, outputs = outputs_with_ids
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return zip(indoml_ids, generated_texts)

def extract_details(text):
    """Extract details from the generated text using regex."""
    pattern = r'supergroup: (.*?), group: (.*?), module: (.*?), brand: (.*?)$'
    match = re.match(pattern, text)
    if match:
        return tuple(item if item is not None else 'na' for item in match.groups())
    return 'na', 'na', 'na', 'na'

def process_dataset(dataset, batch_size=32):
    """Process the entire dataset with batching and parallel execution."""
    model.eval()
    generated_details = []
    
    # Calculate total batches
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, len(dataset), batch_size):
            # Prepare batch data
            batch_data = {
                'indoml_id': dataset['indoml_id'][i:i + batch_size],
                'input_text': dataset['input_text'][i:i + batch_size],
                'retailer_id': dataset['retailer_id'][i:i + batch_size],
                'price': dataset['price'][i:i + batch_size]
            }
                
            # Process batch
            tokenized = tokenize_batch(batch_data)
            generated = generate_text(tokenized)
            decoded = decode_outputs(generated)
                
            # Store results
            for indoml_id, text in decoded:
                generated_details.append((indoml_id, extract_details(text)))
            
            pbar.update(1)
    
    return generated_details

generated_details = process_dataset(test_dataset, batch_size=256)

sorted_generated_details = sorted(generated_details, key=lambda x: x[0])

categories = ['supergroup', 'group', 'module', 'brand']

os.makedirs('../results', exist_ok=True)
result_filename = '../results/final_test_multi-modal-byt5-small.predict'
with open(result_filename, 'w') as file:
    for indoml_id, details in sorted_generated_details:
        result = {"indoml_id": indoml_id}
        for category, value in zip(categories, details):
            result[category] = value
        file.write(json.dumps(result) + '\n')
