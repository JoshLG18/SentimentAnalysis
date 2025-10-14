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

warnings.filterwarnings('ignore')

set_seed() # set the seeds

train_loader, test_loader = prepare_data() # prep the data


#Crearte the LSTM model
class LSTMSentiment(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super(LSTMSentiment, self).__init__()

        self.bert = AutoModel.from_pretrained("ProsusAI/finbert") # uses finbert for the embedding layer

        for param in self.bert.parameters():
            param.requires_grad = False  # freeze FinBERT

        self.lstm = nn.LSTM(input_size=768,  # run the lstm Layers
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)

        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2) # run the attention layer
        self.fc = nn.Linear(hidden_dim * 2, 3) # run the final fully connected layer

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state

        lstm_out, _ = self.lstm(x) # forward pass through the lstm

        att_scores = self.attention(lstm_out) # get the attention scores
        attn_weights = torch.softmax(att_scores, dim=1) # get the attention weights
        context = torch.sum(attn_weights * lstm_out, dim=1) # get the context vector

        out = self.fc(context) # use the context layer to output
        return out


# Initialize model, loss function, and optimizer
model = LSTMSentiment(HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # loss function with label smoothing
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # optim with l2 regularisation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = '../results/saved_models/LSTM.pt' # set where we want to save the model

start_time = time.time() # get the time the training starts
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE)

end_time = time.time() # get the time the training ends

training_time = end_time - start_time # work out how long the model took to train

# save all the metrics and history
save_metrics_and_history(best_metrics, history, training_time)
