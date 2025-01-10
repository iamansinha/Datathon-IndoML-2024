import os
import pandas as pd

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)


train_features = read_jsonl('./datathon_phase_2_data/training_data/train.features')
train_labels = read_jsonl('./datathon_phase_2_data/training_data/train.labels')

train_data = pd.merge(train_features, train_labels, on='indoml_id', how='inner')

train_data.to_json('./train-all.json', orient='records', lines=True)

#------------

columns = ['supergroup', 'group', 'module', 'brand']

# Function to sample n% from each group
def sample_n_percent(group):
    n = 0.1     # 10% split
    return group.sample(frac=n, random_state=42)

# Group by the specified columns and sample 10% from each group
validation_data = train_data.groupby(columns, group_keys=False).apply(sample_n_percent)

# Remove the sampled rows from the original train_data
train_data = train_data.drop(validation_data.index)

# Sort both DataFrames by 'indoml_id' and reset index
train_data = train_data.sort_values(by='indoml_id').reset_index(drop=True)
validation_data = validation_data.sort_values(by='indoml_id').reset_index(drop=True)

os.makedirs('split data', exist_ok=True)

train_data.to_json('./split data/train.json', orient='records', lines=True)
validation_data.to_json('./split data/validation.json', orient='records', lines=True)