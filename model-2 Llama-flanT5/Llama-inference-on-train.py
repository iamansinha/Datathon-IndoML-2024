import pandas as pd
from huggingface_hub import login
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

login(token='YOUR_TOKEN')  # Replace 'YOUR_TOKEN' with your actual huggingface token

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

nrows_train = None
data_path = "../train.json"
train_data = read_jsonl(data_path, nrows=nrows_train)

#---------------------------------------

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

tokenizer.pad_token = tokenizer.eos_token  
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="cuda",
    pad_token_id=tokenizer.eos_token_id
)

def create_messages(row):
    """
    Generates a chat-based format message prompt based on a row of the dataframe.
    Customize this function based on your specific task.
    """
    return [
        {"role": "system", "content": "You are a knowledgeable chatbot who generates product name."},
        {"role": "user", "content": f"""
Product Description: {row['description']}
Price: ${row['price']}
Supergroup: {row['supergroup']}
Group: {row['group']}
Module: {row['module']}
Brand: {row['brand']}

Using the provided details, guess the most accurate and complete product name as per your knowledge of products of all the brands. Ensure that any concatenated words in the description are separated appropriately. The output should consist of one full product name only, without any additional explanation or formatting."""}
    ]

def infer_batch(batch_messages):
    """
    Performs batch inference on a list of chat message prompts.
    """
    outputs = pipeline(batch_messages, max_new_tokens=128, batch_size=len(batch_messages))
    return [output[0]["generated_text"] for output in outputs]

def batched_inference(df, batch_size=32):
    """
    Perform inference on the entire dataframe in batches and add the results as a new column.
    """
    dataloader = DataLoader(df.index, batch_size=batch_size, shuffle=False)

    infer_descs = []

    for batch_indices in tqdm(dataloader, desc="Performing Inference"):
        batch_messages = [create_messages(df.loc[i.item()]) for i in batch_indices]
        batch_outputs = infer_batch(batch_messages)
        infer_descs.extend([output[2]['content'] for output in batch_outputs])

    df["infer_desc"] = infer_descs
    return df

batch_size = 64

train_data_with_inference = batched_inference(train_data, batch_size=batch_size)

columns = ['indoml_id','description','infer_desc','retailer','price']
train_data_features = train_data_with_inference[columns]

filename = 'train_data_features_with_description_inferences.json'
train_data_features.to_json(filename, orient='records', lines=True)