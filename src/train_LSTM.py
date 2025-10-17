# load the libraries
from utils import DEVICE, HIDDEN_DIM, LEARNING_RATE, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import warnings 
import time
from preprocessing import prepare_data
from training_loop import train_model
from utils import save_metrics_and_history
from transformers import AutoModel
from preprocessing import tokenizer

warnings.filterwarnings('ignore')

set_seed() # set the seeds

train_loader, test_loader = prepare_data() # prep the data


#Crearte the LSTM model
class LSTMSentiment(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, dropout=0.3):
        super(LSTMSentiment, self).__init__()

        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")

        for param in self.bert.parameters():
            param.requires_grad = False

        embed_dim = 768

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.bert.parameters())):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        x = bert_out.last_hidden_state

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layernorm(lstm_out)

        att_scores = torch.tanh(self.attention(lstm_out))
        attn_weights = torch.softmax(att_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        out = self.fc(self.dropout(context))
        return out

# Initialize model, loss function, and optimizer
model = LSTMSentiment(HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # loss function with label smoothing
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # optim with l2 regularisation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = '../results/saved_models/LSTM.pt' # set where we want to save the model

start_time = time.time() # get the time the training starts
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE, tokenizer)

end_time = time.time() # get the time the training ends

training_time = end_time - start_time # work out how long the model took to train

# save all the metrics and history
save_metrics_and_history(best_metrics, history, training_time)
