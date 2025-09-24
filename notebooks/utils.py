# utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import random
import numpy as np
import math
from collections import Counter


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
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Irrelevant": 3  # (if it exists in your dataset)
}


# ========== Seed Setting ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
