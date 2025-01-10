# pip install -r requirements.txt
pip install --upgrade pip && pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install numpy pandas transformers datasets evaluate accelerate tqdm scikit-learn huggingface_hub

# Train-Validation splitting of data
unzip phase_2_input_data.zip
python3 data-split.py

# Model-1: ByT5-small fine-tuned on `description` to labels prediction
python3 "./model-1 ByT5-small/training.py"
python3 "./model-1 ByT5-small/inference.py"

# Model-2: Llama + flanT5-base
    # Llama part
python3 "./model-2 Llama-flanT5/Llama-inference-on-train.py"
python3 "./model-2 Llama-flanT5/Llama-lora-ft.py"
python3 "./model-2 Llama-flanT5/ft-Llama-inference-on-test.py"
    # flanT5-base part
python3 "./model-2 Llama-flanT5/flanT5-base-training.py"
python3 "./model-2 Llama-flanT5/flanT5-base-inference.py"

# Model-3: Multi-Modal ByT5-small on 'description', `retailer` and 'price'
python3 "./model-3 MultiModal-ByT5/training.py"
python3 "./model-3 MultiModal-ByT5/inference.py"

# Hallucination correction for all above 3 models
python3 "./label-correction.py"

# Hard-Voting between Model-{1,2,3}
python3 "./hard-voting-with-hierarchy-compliance.py"