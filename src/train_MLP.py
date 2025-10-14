# import all libraries
from utils import DEVICE, HIDDEN_DIM, LEARNING_RATE, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import warnings 
import time
from training_loop import train_model
from preprocessing import prepare_data
from utils import save_metrics_and_history
warnings.filterwarnings('ignore')
from transformers import AutoModel
from preprocessing import tokenizer

set_seed() # sets the seed

train_loader, test_loader = prepare_data() # prepares all the data

# define the mlp architecture
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim=3):
        super(MLP, self).__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")  # FinBERT as embedding layer

        # Freeze all FinBERT parameters for efficiency
        for param in self.bert.parameters():
            param.requires_grad = False

        # Layer 1
        self.fc1 = nn.Linear(768, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, input_ids, attention_mask):
        # Extract contextual embeddings from FinBERT
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling to get a single sentence embedding
        x = torch.mean(bert_output.last_hidden_state, dim=1)

        # Forward pass through MLP layers
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)  # output raw logits
        return out
    
# Initialize model, loss function, and optimiser
model = MLP(
            hidden_dim=HIDDEN_DIM,
            output_dim=3,
            ).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # L2 Regularisation
# use reduce LR on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5) 

model_save_loc = '../results/saved_models/mlp.pt' # set the location to save the model

start_time = time.time() # get the time the training starts
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE, tokenizer)

end_time = time.time() # get the time the training ends

training_time = end_time - start_time # work out how long the model is training for

# save the metrics and history of the best model
save_metrics_and_history(best_metrics, history, training_time)
