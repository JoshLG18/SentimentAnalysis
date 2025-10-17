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
    # define the output dim to 3 as there are 3 classes
    # dropout to 0.3 sop that 30% of neurons are switched off
    def __init__(self, hidden_dim, num_layers=3, dropout=0.3):
        super(LSTMSentiment, self).__init__()

        self.bert = AutoModel.from_pretrained("ProsusAI/finbert") # load in the finbert model

        # Freeze all FinBERT parameters for trianing efficiency - would train for so long otherwise
        for param in self.bert.parameters(): # loops through all parameters in the bert model
            param.requires_grad = False # sets all parameters no gradient

        inp_dim = 768 # sets the embedding dim to the ouput dimension of finbert

        self.lstm = nn.LSTM( # defines the lstm with 3 layers and bidirectional
            input_size=inp_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # normalises the activations after lstm output - stops exploding gradients
        self.layernorm = nn.LayerNorm(hidden_dim * 2) 

        # defines the attention layer - maps each hidden state to a learned attention score
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2) 

        # definees the dropout - randomly turns off neurons
        self.dropout = nn.Dropout(dropout)

        # final layer to classify into the 3 classes
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, input_ids, attention_mask):
        # Extract contextual embeddings from FinBERT
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        x = bert_output.last_hidden_state

        # pass the token embeddings into the lstm
        lstm_out, _ = self.lstm(x) # captures sequential dependencies in both directions

        # Normalise LSTM outputs to stabilise training and reduce covariate shift
        lstm_out = self.layernorm(lstm_out)

        # Compute attention scores for each time step
        att_scores = torch.tanh(self.attention(lstm_out)) # non-linear transformation of LSTM outputs

        # Convert attention scores to weights using softmax
        attn_weights = torch.softmax(att_scores, dim=1) # ensures weights sum to 1

        # Compute context vector as the weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Apply dropout for regularisation and pass through final classifier
        out = self.fc(self.dropout(context))

        return out # outputs logits for 3 sentiment classes

# Initialize model, loss function, and optimizer
model = LSTMSentiment(HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to improve regularisation / decrease overfitting
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # optim with l2 regularisation
# if the loss hasnt improved after 2 epochs reduce lr by 50%
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = '../results/saved_models/LSTM.pt' # set where we want to save the model

start_time = time.time() # get the time the training starts
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE, tokenizer)

end_time = time.time() # get the time the training ends

training_time = end_time - start_time # work out how long the model took to train

# save all the metrics and history
save_metrics_and_history(best_metrics, history, training_time)
