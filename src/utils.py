# utils.py

import torch
import random
import numpy as np

# ========== Device ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== Hyperparameters ==========
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 300
LEARNING_RATE = 0.005
LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "irrelevant": 3  # (if it exists in your dataset)
}


# ========== Seed Setting ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
