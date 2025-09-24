
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from utils import EMBEDDING_DIM, MAX_LEN, BATCH_SIZE, LABEL_MAPPING


# ========== Load and Split Data ==========
def load_data(train_loc, val_loc, test_size=0.2, random_state=123):
    """Load CSVs, clean, and split into train/test pandas DataFrames."""
    raw_train_data = pd.read_csv(train_loc)
    raw_val_data = pd.read_csv(val_loc)

    raw_train_data.columns = ["ID", "Object", "Sentiment", "Tweet"]
    raw_val_data.columns = ["ID", "Object", "Sentiment", "Tweet"]

    # Combine
    full_data = pd.concat([raw_train_data, raw_val_data])

    # Clean
    full_data = full_data.drop_duplicates()
    full_data = full_data.drop(columns=["ID", "Object"])
    full_data = full_data.rename(columns={"Tweet": "text", "Sentiment": "label"})
    full_data["text"] = full_data["text"].astype(str)
    # normalise labels to lowercase strings and drop anything unexpected
    full_data["label"] = full_data["label"].astype(str).str.strip().str.lower()
    full_data = full_data[full_data["label"].isin(LABEL_MAPPING.keys())]


    # Train/test split
    train_df, test_df = train_test_split(full_data, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ========== Tokenizer & GloVe ==========
tokenizer = get_tokenizer("basic_english") # defines the tokenizer which splits sentences into indivdual words
glove = GloVe(name="6B", dim=EMBEDDING_DIM)


def yield_glove_tokens(data_iter):
    """Yield tokens that exist in GloVe vocab (for vocab building)."""
    for _, row in data_iter.iterrows(): # loops though all the rows in the training df and tokenises the tweet
        tokens = tokenizer(row["text"])
        yield [t for t in tokens if t in glove.stoi] # only keeps tokens that exist in the glove dictionary


def build_vocab(train_df): # builds a mapping between words and ids
    """Build vocab restricted to tokens in GloVe embeddings."""
    vocab = build_vocab_from_iterator(yield_glove_tokens(train_df), specials=["<pad>", "<unk>"]) # adds special tokens e.g. pad to make sure all sentences are the same length. unk = unknown word
    vocab.set_default_index(vocab["<unk>"]) # sets the default index to unknown
    return vocab


# ========== Preprocessing ==========
def preprocess(example, vocab=None): # tokenises the text - converts each token into interger from the vocab
    text = str(example["text"])
    tokens = tokenizer(text)
    input_ids = vocab(tokens)[:MAX_LEN]
    label_str = str(example["label"]).strip().lower()
    label = LABEL_MAPPING[label_str]          # <-- no int() here
    return {"input_ids": input_ids, "label": label}


# ========== Collate Function ==========
def collate_batch(batch):
    texts = [torch.tensor(sample["input_ids"], dtype=torch.int64) for sample in batch]
    labels = [sample["label"] for sample in batch]  # already int
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)  # shape (batch_size,)
    return texts_padded, labels

# ========== Embedding Matrix ==========
def build_embedding_matrix(vocab, embedding_dim=EMBEDDING_DIM):
    # creates an embedding matrix 
    # where each row corresponds to a word in the vocab
    # if the word exists in Glove include its embedding, if not assign random embedding
    embedding_matrix = torch.zeros(len(vocab), embedding_dim)
    for idx, token in enumerate(vocab.get_itos()):
        if token in glove.stoi:
            embedding_matrix[idx] = glove[token]
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.6
    return embedding_matrix # returns a matrix that can be fed directly into the embedding layer, speeds up training


# ========== Main Setup Pipeline ==========
def prepare_data(train_path, val_path):
    """
    Full pipeline:
    - Load data
    - Build vocab
    - tokenise the text
    - Apply preprocessing
    - Return DataLoaders + embedding matrix
    """
    # Load dataframes
    train_df, test_df = load_data(train_path, val_path)

    # Build vocab
    global vocab
    vocab = build_vocab(train_df)

    # Convert pandas -> HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ===== in prepare_data(): pass vocab explicitly to map =====
    train_dataset = train_dataset.map(lambda ex: preprocess(ex, vocab))
    test_dataset  = test_dataset.map(lambda ex: preprocess(ex, vocab))


    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "label"])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    # Embeddings
    embedding_matrix = build_embedding_matrix(vocab)

    return train_loader, test_loader, vocab, embedding_matrix