from utils import DEVICE, EMBEDDING_DIM, HIDDEN_DIM,MAX_LEN, LEARNING_RATE, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import warnings 
import time

from training_loop import train_model
from preprocessing import prepare_data
from utils import save_metrics_and_history
warnings.filterwarnings('ignore')
set_seed()

train_path = './data/twitter_training.csv'
test_path = './data/twitter_validation.csv'
train_loader, test_loader, vocab, embedding_matrix = prepare_data(train_path, test_path)

#Create the Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len, dropout=0.3):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=vocab["<pad>"]) # embedding layer
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True 
        ) # Transformer Encoder Layer

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Transformer Encoder

        self.classifier = nn.Sequential( # Final classification layer
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        # Pad mask from token IDs
        mask = (x == vocab["<pad>"])  # (batch, seq_len)

        # Embed + positional encoding
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x)
        x = x + self.pos_embedding(positions)

        x = self.transformer_encoder(x, src_key_padding_mask=mask) # Transformer Encoder

        x = x.mean(dim=1)  # Global average pooling

        return self.classifier(x)

# Initialize model, loss function, and optimizer
model = Transformer(
    vocab_size=len(vocab),
    embed_dim=EMBEDDING_DIM,
    num_heads=4,  
    hidden_dim=HIDDEN_DIM,
    num_layers=2,
    max_len=MAX_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = './results/saved_models/transformer.pt'

start_time = time.time() 
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE)

end_time = time.time()

training_time = end_time - start_time
# save the metrics and history of the best model
save_metrics_and_history(best_metrics, history, training_time)