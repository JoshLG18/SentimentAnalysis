import torch
import random
import numpy as np
import os, sys, json
from datetime import datetime

# ========== Device ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== Hyperparameters ==========
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 100
MAX_LEN = 300
LEARNING_RATE = 1e-3
LABEL_MAPPING = { # set the label mapping for the dataset
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "irrelevant": 3
}

# ========== Seed Setting ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_metrics_and_history(best_metrics, history,training_time):
    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # File to store all results together
    results_file = "../results/all_results.json"

    # Get current script filename (e.g. train_LSTM.py or train_MLP.py)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Add timestamp + script name into the JSON content
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": run_time,
        "training_time": training_time,
        "best_metrics": best_metrics,
        "history": history
    }

    # Load existing results if they exist
    if os.path.exists(results_file): # checks if the file exists
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {} # if not create the dictionary

    # Update only this model’s entry
    all_results[script_name] = entry

    # Save back
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Saved metrics & history for {script_name} at {run_time}")
        