# load libraries
from utils import DEVICE, HIDDEN_DIM, LEARNING_RATE, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import time
from training_loop import train_model
from preprocessing import prepare_data
from utils import save_metrics_and_history
from transformers import AutoModel
from preprocessing import tokenizer

warnings.filterwarnings('ignore') # turn off warnings so i don't lose my mind
set_seed()

# Load processed data
train_loader, test_loader = prepare_data()

# Define Transformer Model Using FinBERT
class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=4, dropout=0.3):
        super(Transformer, self).__init__()

        self.bert = AutoModel.from_pretrained("ProsusAI/finbert") # loading pre trained finbert for embeddings

        # Freeze all FinBERT parameters for trianing efficiency - would train for so long otherwise
        for param in self.bert.parameters(): # loops through all parameters in the bert model
            param.requires_grad = False # sets all parameters no gradient

        embed_dim = 768  # FinBERT output dimension

        self.layernorm = nn.LayerNorm(embed_dim) # define the normalisation layer

        encoder_layer = nn.TransformerEncoderLayer( # define the transformer encoder
            d_model=embed_dim,
            nhead=num_heads, # number of self attention heads
            dim_feedforward=hidden_dim, # size of the feedforward sub-layer inside the encoder
            dropout=dropout,
            activation='relu', # define the activation function - relu most common
            batch_first=True
        )

        # create the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_norm = nn.LayerNorm(embed_dim)

        # use a simple 2 layer FCL to output the logits
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, input_ids, attention_mask):
        # get the embeddings from finbert
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        x = bert_out.last_hidden_state # get the output of finbert

        x = self.layernorm(x) # add a normalisation layer for finbert encodings - stop exploding gradients

        # Create mask so attention ignores padding tokens
        src_key_padding_mask = ~attention_mask.bool()

        # Pass through the transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = x.mean(dim=1)  # Global average pooling - summarise all tokens to a sentence level

        x = self.final_norm(x) # add a normalisation layer

        return self.classifier(x) # run the classifier to get the output and return it


# Initialise the model, loss function, optimiser and scheduler
model = Transformer(hidden_dim=HIDDEN_DIM).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to improve regularisation / decrease overfitting
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # L2 reg
# if the loss hasnt improved after 2 epochs reduce lr by 50%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5) # decrease LR on plateau

model_save_loc = '../results/saved_models/transformer.pt'

start_time = time.time() # get the time before training
history, best_metrics = train_model( # train the model
    model_save_loc,
    model,
    train_loader,
    test_loader,
    optimiser,
    scheduler,
    criterion,
    DEVICE,
    tokenizer
)
end_time = time.time() # get the time at the end of training
training_time = end_time - start_time # work out how long the model was training for

# save the metrics and history
save_metrics_and_history(best_metrics, history, training_time)