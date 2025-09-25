
from utils import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, MAX_LEN, LEARNING_RATE, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import warnings 
import time
from preprocessing import prepare_data
from training_loop import train_model
from utils import save_metrics_and_history

warnings.filterwarnings('ignore')

set_seed()

train_path = './data/twitter_training.csv'
test_path = './data/twitter_validation.csv'
train_loader, test_loader, vocab, embedding_matrix = prepare_data(train_path, test_path)


#Crearte the LSTM model
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(LSTMSentiment, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=vocab["<pad>"]) # embedding layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True) # LSTM layer
        self.attention = nn.Linear(hidden_dim*2, hidden_dim*2) # Attention layer
        self.fc = nn.Linear(hidden_dim*2, 4)
        
    def forward(self, x):
        x = self.embedding(x) # run the embedding layer
        lstm_out, _ = self.lstm(x) # run lstm layer

        att_scores = self.attention(lstm_out) # compute attention scores
        attn_weights = torch.softmax(att_scores, dim=1) # normalize scores to weights

        context = torch.sum(attn_weights * lstm_out, dim=1) # weighted sum to get context vector

        out = self.fc(context)
        return out.squeeze()

# Initialize model, loss function, and optimizer
model = LSTMSentiment(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = './results/saved_models/LSTM.pt'

start_time = time.time() 
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE)

end_time = time.time()

training_time = end_time - start_time

save_metrics_and_history(best_metrics, history, training_time)
