import pandas as pd
import json
import difflib
from tqdm.auto import tqdm
import os
import glob

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)

# Function to find the closest valid label using difflib
def find_closest_label(predicted_label, valid_labels):
    closest_match = difflib.get_close_matches(predicted_label, valid_labels, n=1, cutoff=0.0)
    return closest_match[0] if closest_match else None

# Function to check if a prediction is valid based on hierarchical structure
def check_prediction(row, hierarchy_to_groups, hierarchy_to_modules, hierarchy_to_brands):
    supergroup = row['supergroup']
    group = row['group']
    module = row['module']
    brand = row['brand']
    
    # Validate each hierarchical level based on the higher levels
    valid_groups = hierarchy_to_groups.get(supergroup, set())
    valid_modules = hierarchy_to_modules.get((supergroup, group), set())
    valid_brands = hierarchy_to_brands.get((supergroup, group, module), set())
    
    return (
        supergroup in hierarchy_to_groups and
        group in valid_groups and
        module in valid_modules and
        brand in valid_brands
    )

# Function to correct hallucinated predictions using hierarchical structure for brand
def correct_hallucinated_prediction(row, hierarchy_to_groups, hierarchy_to_modules, hierarchy_to_brands):
    corrected_row = row.copy()
    corrected = False  # Flag to track if any label was corrected
    
    # Correct hierarchical labels
    if row['supergroup'] not in hierarchy_to_groups:
        corrected_row['supergroup'] = find_closest_label(row['supergroup'], hierarchy_to_groups.keys())
        corrected = True
    
    valid_groups = hierarchy_to_groups.get(corrected_row['supergroup'], set())
    if corrected_row['group'] not in valid_groups:
        corrected_row['group'] = find_closest_label(corrected_row['group'], valid_groups)
        corrected = True
    
    valid_modules = hierarchy_to_modules.get((corrected_row['supergroup'], corrected_row['group']), set())
    if corrected_row['module'] not in valid_modules:
        corrected_row['module'] = find_closest_label(corrected_row['module'], valid_modules)
        corrected = True
    
    # Correct brand using the valid brands for the specific (supergroup, group, module)
    hierarchy_key = (corrected_row['supergroup'], corrected_row['group'], corrected_row['module'])
    valid_brands_for_hierarchy = hierarchy_to_brands.get(hierarchy_key, set())
    
    if corrected_row['brand'] not in valid_brands_for_hierarchy:
        corrected_row['brand'] = find_closest_label(corrected_row['brand'], valid_brands_for_hierarchy)
        corrected = True
    
    return corrected_row, corrected

# Load training data
train_data = read_jsonl("./train-all.json")

# Create a dictionary mapping (supergroup, group, module) to valid brands
hierarchy_to_groups = train_data.groupby('supergroup')['group'].apply(set).to_dict()
hierarchy_to_modules = train_data.groupby(['supergroup', 'group'])['module'].apply(set).to_dict()
hierarchy_to_brands = train_data.groupby(['supergroup', 'group', 'module'])['brand'].apply(set).to_dict()

# Process each .predict file in the ./result folder
for filepath in glob.glob('./results/*.predict'):
    # Load test data
    test_predicts = read_jsonl(filepath)

    hallucinated_predictions = []
    corrected_predictions = []
    correction_count = 0  # Counter for corrected rows

    for index, row in tqdm(test_predicts.iterrows(), total=len(test_predicts)):
        if not check_prediction(row, hierarchy_to_groups, hierarchy_to_modules, hierarchy_to_brands):
            hallucinated_predictions.append(row.to_dict())  # Store hallucinated predictions
            corrected_row, corrected = correct_hallucinated_prediction(row, hierarchy_to_groups, hierarchy_to_modules, hierarchy_to_brands)
            corrected_predictions.append(corrected_row.to_dict())  # Store corrected predictions
            
            if corrected:  # Increment correction count if the row was corrected
                correction_count += 1
        else:
            corrected_predictions.append(row.to_dict())  # Valid rows don't need correction

    # Generate the output file path
    os.makedirs('./results/corrected results', exist_ok=True)
    output_filename = f'{filepath.split('/')[-1].split('.')[0]}-corrected.predict'
    output_path = f"./results/corrected results/{output_filename}"

    # Write corrected predictions to the output file
    with open(output_path, 'w') as f:
        for corrected_prediction in corrected_predictions:
            f.write(json.dumps({
                "indoml_id": corrected_prediction["indoml_id"],
                "supergroup": corrected_prediction["supergroup"],
                "group": corrected_prediction["group"],
                "module": corrected_prediction["module"],
                "brand": corrected_prediction["brand"]
            }) + '\n')

    print(f"{output_filename} saved with {(correction_count/len(test_predicts))*100:.2f}% rows corrected.")

