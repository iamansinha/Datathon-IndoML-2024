import os
import pandas as pd
from collections import Counter, defaultdict
import glob
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

# Function to read JSONL files efficiently with error handling
def read_jsonl(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Reads a JSON lines file into a DataFrame."""
    try:
        return pd.read_json(file_path, lines=True, nrows=nrows)
    except ValueError as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# Load predictions from multiple JSONL files
def load_predictions(file_paths: List[str]) -> List[pd.DataFrame]:
    """Loads predictions from multiple JSONL files into a list of DataFrames."""
    return [read_jsonl(fp) for fp in file_paths if read_jsonl(fp).shape[0] > 0]

# Build hierarchy with fallback defaults from the training data
def build_hierarchy_with_fallback(train_data: pd.DataFrame):
    """Builds hierarchical mappings with fallback defaults from train data."""
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    frequency_counters = {
        "group": defaultdict(Counter),
        "module": defaultdict(Counter),
        "brand": defaultdict(Counter)
    }

    # Populate the hierarchy and track frequency of child elements
    for _, row in train_data.iterrows():
        # Build the hierarchy structure
        hierarchy[row["supergroup"]][row["group"]][row["module"]].add(row["brand"])

        # Track frequency of each child element for fallback purposes
        frequency_counters["group"][row["supergroup"]][row["group"]] += 1
        frequency_counters["module"][row["group"]][row["module"]] += 1
        frequency_counters["brand"][row["module"]][row["brand"]] += 1

    # Create fallback dictionaries by selecting the most frequent child for each parent
    most_frequent = {
        "group": {sg: counter.most_common(1)[0][0] for sg, counter in frequency_counters["group"].items()},
        "module": {g: counter.most_common(1)[0][0] for g, counter in frequency_counters["module"].items()},
        "brand": {m: counter.most_common(1)[0][0] for m, counter in frequency_counters["brand"].items()}
    }

    return hierarchy, most_frequent

# Perform hard voting for a specific label across multiple models
def hard_vote(predictions: List[Dict[str, Any]], label: str) -> str:
    """Applies hard voting to determine the most common label."""
    votes = Counter(pred[label] for pred in predictions)
    return votes.most_common(1)[0][0]

# Hierarchical voting with fallback handling
def hierarchical_vote_with_fallback(
    predictions: List[Dict[str, Any]], 
    hierarchy: Dict, 
    fallback: Dict[str, defaultdict]
) -> Dict[str, str]:
    """Performs hierarchical voting with fallback to most frequent values."""
    supergroup = hard_vote(predictions, "supergroup")

    # Handle group voting with fallback
    valid_groups = [pred["group"] for pred in predictions if pred["group"] in hierarchy[supergroup]]
    group = Counter(valid_groups).most_common(1)[0][0] if valid_groups else fallback["group"][supergroup]

    # Handle module voting with fallback
    valid_modules = [pred["module"] for pred in predictions if pred["module"] in hierarchy[supergroup][group]]
    module = Counter(valid_modules).most_common(1)[0][0] if valid_modules else fallback["module"][group]

    # Handle brand voting with fallback
    valid_brands = [pred["brand"] for pred in predictions if pred["brand"] in hierarchy[supergroup][group][module]]
    brand = Counter(valid_brands).most_common(1)[0][0] if valid_brands else fallback["brand"][module]

    return {
        "supergroup": supergroup,
        "group": group,
        "module": module,
        "brand": brand
    }

# Combine predictions using hierarchical voting with fallback
def combine_predictions_with_fallback(
    all_predictions: List[pd.DataFrame], 
    hierarchy: Dict, 
    fallback: Dict[str, defaultdict]
) -> List[Dict[str, Any]]:
    """Combines predictions using hierarchical voting with fallback."""
    num_data = len(all_predictions[0])
    combined_results = []

    for i in tqdm(range(num_data), desc="Combining predictions"):
        # Collect predictions for the current data point across all models
        sample_predictions = [model.iloc[i].to_dict() for model in all_predictions]

        # Create the final prediction with hierarchical voting and fallback
        final_prediction = {
            "indoml_id": int(sample_predictions[0]["indoml_id"]),  # Ensure native Python int
            **hierarchical_vote_with_fallback(sample_predictions, hierarchy, fallback)
        }
        combined_results.append(final_prediction)

    return combined_results

# Load prediction files
prediction_files = glob.glob('./results/corrected results/*.predict')

if not prediction_files:
    raise FileNotFoundError("No prediction files found in the specified directory.")

all_predictions = load_predictions(prediction_files)

# Ensure all prediction files have the same length
if not all(len(pred) == len(all_predictions[0]) for pred in all_predictions):
    raise ValueError("Mismatch in prediction lengths across models.")

# Load training data to build the hierarchy and fallback defaults
train_data = read_jsonl("./train-all.json")
hierarchy, fallback = build_hierarchy_with_fallback(train_data)

# Combine predictions using hierarchical voting with fallback
final_predictions = combine_predictions_with_fallback(all_predictions, hierarchy, fallback)

# Save the final predictions to a JSONL file
os.makedirs('./results/final result', exist_ok=True)
result_filename = './results/final result/final_test_hard-voting-3-models.predict'
pd.DataFrame(final_predictions).to_json(result_filename, orient='records', lines=True)
print(f"Final predictions saved to {result_filename}")
