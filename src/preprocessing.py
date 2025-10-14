import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from utils import BATCH_SIZE

# Constants
MODEL_NAME = "ProsusAI/finbert" # defines the finbert model used

# Load tokenizer and collator
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load and process dataset
def prepare_data(test_size=0.2, random_state=123):
    # Load raw data from Hugging Face
    raw_dataset = load_dataset("lukecarlate/english_finance_news")
    df = raw_dataset["train"].to_pandas()[["newscontents", "label"]]
    df.columns = ["text", "label"]

    # Train/test split using stratified sampling - ensures class imbalance is maintained in the splits
    train_df, test_df = train_test_split(
        df,
        test_size=test_size, # defines the test size as 20%
        random_state=random_state, # ensures random splits are reproduceable
        stratify=df["label"]  # ensures balanced label proportions
        )

    train_ds = Dataset.from_pandas(train_df) # converts the pandas dfs into huggingface
    test_ds = Dataset.from_pandas(test_df) # converts the pandas dfs into huggingface

    # Tokenisation
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding=False,max_length=64)

    train_ds = train_ds.map(tokenize, batched=True) # tokenises the dataset using finbert
    test_ds = test_ds.map(tokenize, batched=True) # tokenises the dataset using finbert

    # converts the hugging face datasets into tensors
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"]) 
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    return train_loader, test_loader # returns the loaders to be used in the training scripts
