from utils import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, set_seed
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

set_seed()

train_loader, test_loader = prepare_data()


class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")

        for param in self.bert.parameters():
            param.requires_grad = False  # freeze FinBERT

        self.fc1 = nn.Linear(768, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.mean(bert_output.last_hidden_state, dim=1)  # mean pooling
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Initialize model, loss function, and optimiser
model = MLP(
            hidden_dim=HIDDEN_DIM,
            output_dim=3,
            )
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)

model_save_loc = '../results/saved_models/mlp.pt'

start_time = time.time() 
# train the model
history, best_metrics = train_model(model_save_loc, model, train_loader, test_loader, optimiser,scheduler, criterion, DEVICE)

end_time = time.time()

training_time = end_time - start_time
# save the metrics and history of the best model
save_metrics_and_history(best_metrics, history, training_time)
