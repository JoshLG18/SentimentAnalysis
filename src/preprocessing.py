
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

    # Normalise labels to lowercase strings and drop anything unexpected
    full_data["label"] = full_data["label"].astype(str).str.strip().str.lower() # make sure label is str, strip the lable and make it lower
    full_data = full_data[full_data["label"].isin(LABEL_MAPPING.keys())] # assign the label a mapping 


    # Train/test split
    train_df, test_df = train_test_split(full_data, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

#  get tokeniser and glove
tokeniser = get_tokenizer("basic_english") # defines the tokenizer which splits sentences into indivdual words
glove = GloVe(name="6B", dim=EMBEDDING_DIM)


def create_glove_tokens(data_iter):
    for _, row in data_iter.iterrows(): # loops though all the rows in the training df and tokenises the tweet
        tokens = tokeniser(row["text"]) # tokenise the tweet
        yield [t for t in tokens if t in glove.stoi] # only keeps tokens that exist in the glove dictionary


def build_vocab(train_df): # builds a mapping between words and ids
    vocab = build_vocab_from_iterator(create_glove_tokens(train_df), specials=["<pad>", "<unk>"]) # adds special tokens e.g. pad to make sure all sentences are the same length. unk = unknown word
    vocab.set_default_index(vocab["<unk>"]) # sets the default index to unknown
    return vocab

#  processing 
def preprocess(example, vocab=None): # tokenises the text - converts each token into interger from the vocab
    text = str(example["text"])
    tokens = tokeniser(text) # tokenise text
    input_ids = vocab(tokens)[:MAX_LEN]
    label_str = str(example["label"]).strip().lower() # make sure str, strip the label and make lower
    label = LABEL_MAPPING[label_str]  
    return {"input_ids": input_ids, "label": label}

def make_batches(batch):
    texts, labels = [], []
    for sample in batch:
        # Ensure non-empty input_ids
        input_ids = sample["input_ids"]
        if len(input_ids) == 0:
            input_ids = [vocab["<unk>"]]  # fallback: unknown token

        texts.append(torch.tensor(input_ids, dtype=torch.long))
        labels.append(int(sample["label"]))  # force int

    # Pad
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)

    return texts_padded, labels

#  create embedding matrix 
def create_embedding_matrix(vocab, embedding_dim=EMBEDDING_DIM):
    # creates an embedding matrix 
    # where each row corresponds to a word in the vocab
    # if the word exists in Glove include its embedding, if not assign random embedding
    embedding_matrix = torch.zeros(len(vocab), embedding_dim)
    for idx, token in enumerate(vocab.get_itos()):
        if token in glove.stoi: # check if the word exists in the glove dictionary
            embedding_matrix[idx] = glove[token] # assign the word with a glove token
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1 # if the word is unknown give it a random value
    return embedding_matrix # returns a matrix that can be fed directly into the embedding layer, speeds up training


#  Processing Pipeline 
def prepare_data(train_path, val_path):
    # Load dataframes
    train_df, test_df = load_data(train_path, val_path)

    # Build vocab
    global vocab
    vocab = build_vocab(train_df)

    # Convert pandas -> HuggingFace Dataset so can be passed to transformers later
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Map vocab to indexes
    train_dataset = train_dataset.map(lambda ex: preprocess(ex, vocab))
    test_dataset  = test_dataset.map(lambda ex: preprocess(ex, vocab))

    print(set(train_dataset["label"]))

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "label"])
  
    # create the dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_batches
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=make_batches
    )

    # create the embedding matrix
    embedding_matrix = create_embedding_matrix(vocab)

    return train_loader, test_loader, vocab, embedding_matrix