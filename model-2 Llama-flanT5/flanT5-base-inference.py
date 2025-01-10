import re
import json
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_test = None
test_data = read_jsonl('../phase_2_input_data/final_test_data/final_test_data.features', nrows=nrows_test)

def preprocess_data(data):
    data['input_text'] = data.apply(lambda row: f"prod{row['infer_desc']}", axis=1)
    return data[['indoml_id', 'input_text']]

test_processed = preprocess_data(test_data)
test_dataset = Dataset.from_pandas(test_processed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'flan-t5-base-ft-final'
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()


def tokenize_batch(id_and_inputs_dict):
    """Tokenize the input text."""
    indoml_ids = id_and_inputs_dict['indoml_id']
    inputs = id_and_inputs_dict['input_text']
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return indoml_ids, inputs


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


def clean_repeated_patterns(text):
    """Clean repeated patterns in the text."""
    cleaned_data = text.split(' brand')[0] 
    return cleaned_data


batch_size = 128
generated_details = []

# Separate queues for tokenization, generation, and decoding
tokenization_futures = []
generation_futures = []
decoding_futures = []

# Create a progress bar for batch completion
total_batches = (test_dataset.num_rows + batch_size - 1) // batch_size
with tqdm(total=total_batches, desc="Processing batches") as pbar:
    # Thread pool to parallelize tokenization, generation, and decoding
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i in range(0, test_dataset.num_rows, batch_size):
            # Get the current batch inputs
            batch_inputs = test_dataset[i:i + batch_size]

            # Submit tokenization as a separate task to the CPU thread
            tokenization_future = executor.submit(tokenize_batch, batch_inputs)
            tokenization_futures.append(tokenization_future)

            # If there's a tokenized batch ready, submit generation for that batch to the GPU
            if len(tokenization_futures) > 0 and tokenization_futures[0].done():
                tokenized_inputs = tokenization_futures.pop(0).result()
                generation_future = executor.submit(generate_text, tokenized_inputs)
                generation_futures.append(generation_future)

            # If there's a generated batch ready, submit decoding for that batch to the CPU
            if len(generation_futures) > 0 and generation_futures[0].done():
                generated_outputs = generation_futures.pop(0).result()
                decoding_future = executor.submit(decode_outputs, generated_outputs)
                decoding_futures.append(decoding_future)

            # If there's a decoded batch ready, extract details from the decoded texts
            if len(decoding_futures) > 0 and decoding_futures[0].done():
                decoded_texts = decoding_futures.pop(0).result()
                for indoml_id, generated_text in decoded_texts:
                    generated_details.append((indoml_id, extract_details(generated_text)))
                pbar.update(1)  # Update the progress bar when a batch is fully processed

        # Handle any remaining futures at the end
        # Process remaining tokenization futures
        while tokenization_futures:
            tokenized_inputs = tokenization_futures.pop(0).result()
            generation_future = executor.submit(generate_text, tokenized_inputs)
            generation_futures.append(generation_future)

        # Process remaining generation futures
        while generation_futures:
            generated_outputs = generation_futures.pop(0).result()
            decoding_future = executor.submit(decode_outputs, generated_outputs)
            decoding_futures.append(decoding_future)

        # Process remaining decoding futures
        while decoding_futures:
            decoded_texts = decoding_futures.pop(0).result()
            for indoml_id, generated_text in decoded_texts:
                generated_details.append((indoml_id, extract_details(generated_text)))
            pbar.update(1)  # Update progress for the last batches

sorted_generated_details = sorted(generated_details, key=lambda x: x[0])

categories = ['supergroup', 'group', 'module', 'brand']

os.makedirs('../results', exist_ok=True)
result_filename = '../results/final_test_Llama-flanT5.predict'
with open(result_filename, 'w') as file:
    for indoml_id, details in sorted_generated_details:
        result = {"indoml_id": indoml_id}
        for category, value in zip(categories, details):
            result[category] = value
        file.write(json.dumps(result) + '\n')
