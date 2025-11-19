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
warnings.filterwarnings('ignore') # turn off warnings so i don't lose my mind
from transformers import AutoModel
from preprocessing import tokenizer

set_seed() # sets the seed

train_loader, test_loader = prepare_data() # prepares all the data

# define the mlp architecture
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim=3): # define the output dim to 3 as there are 3 classes
        super(MLP, self).__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")  # load finbert to be the embedding layer

        # Freeze all FinBERT parameters for trianing efficiency - would train for so long otherwise
        for param in self.bert.parameters(): # loops through all parameters in the bert model
            param.requires_grad = False # sets all parameters no gradient

        # Layer 1
        self.fc1 = nn.Linear(768, hidden_dim) # define the linear layer with 768 input and hidden dim neurons - z1 = xW + b
        self.relu1 = nn.ReLU() # define the activation function - a1 = activation(z1)
        self.dropout1 = nn.Dropout(0.3) # define the dropout layer - applys a mask to make some activations 0

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # define the linear layer with hidden dim and hidden dim neurons / 2 - z3 = a1W + b
        self.relu2 = nn.ReLU() # define the activation function - a2 = activation(z2)
        self.dropout2 = nn.Dropout(0.3) # define the dropout layer - applys a mask to make some activations 0

        # Layer 3
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # define the linear layer with HD / 2 and HD / 4 neurons - z3 = a2W + b
        self.relu3 = nn.ReLU() # define the activation function - a3 = activation(z3)
        self.dropout3 = nn.Dropout(0.3) # define the dropout layer - applys a mask to make some activations 0

        # Layer Normalisation - stabilises feature distributions
        self.layernorm = nn.LayerNorm(hidden_dim // 4) 
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim) # define the linear layer with HD / 4 and output dim neurons - z = a3W + b


    def forward(self, input_ids, attention_mask):
        # Extract contextual embeddings from FinBERT
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling to get a single sentence embedding
        x = torch.mean(bert_output.last_hidden_state, dim=1) # aggregates token level emeddings to sentence level

        # Forward pass through hidden layers
        x = self.dropout1(self.relu1(self.fc1(x))) # Linear -> ReLU -> Dropout
        x = self.dropout2(self.relu2(self.fc2(x))) # Linear -> ReLU -> Dropout
        x = self.dropout3(self.relu3(self.fc3(x))) # Linear -> ReLU -> Dropout

        x = self.layernorm(x) # layer normalisation - stops exploding gradients

        # Output logits
        out = self.fc4(x) # run the lienar layer to get outputs

        return out

# Initialize model, loss function, and optimiser
model = MLP(
            hidden_dim=HIDDEN_DIM,
            output_dim=3,
            ).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to improve regularisation / decrease overfitting
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # L2 Regularisation
# if the loss hasnt improved after 2 epochs reduce lr by 50%
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5) 

model_save_loc = '../results/saved_models/mlp.pt' # set the location to save the model

start_time = time.time() # get the time the training starts

# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE, tokenizer)

end_time = time.time() # get the time the training ends

training_time = end_time - start_time # work out how long the model is training for

# save the metrics and history of the best model
save_metrics_and_history(best_metrics, history, training_time)


# References:
# https://docs.pytorch.org/docs/stable/index.html