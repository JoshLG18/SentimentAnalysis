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

warnings.filterwarnings('ignore')
set_seed()

# Load processed data
train_loader, test_loader, tokenizer = prepare_data()


# === Define Transformer Model Using FinBERT ===
class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.3):
        super(Transformer, self).__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze FinBERT

        embed_dim = 768  # FinBERT output dimension

        self.layernorm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state  # [batch, seq_len, 768]

        x = self.layernorm(x)

        # Invert attention_mask: Transformer expects True for padding tokens
        src_key_padding_mask = ~attention_mask.bool()
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = x.mean(dim=1)  # Global average pooling

        return self.classifier(x)


# === Initialise, Train, and Save ===
model = Transformer(hidden_dim=HIDDEN_DIM).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = '../results/saved_models/transformer.pt'

start_time = time.time()
history, best_metrics = train_model(
    model_save_loc,
    model,
    train_loader,
    test_loader,
    optimiser,
    scheduler,
    criterion,
    DEVICE
)
end_time = time.time()
training_time = end_time - start_time

save_metrics_and_history(best_metrics, history, training_time)
