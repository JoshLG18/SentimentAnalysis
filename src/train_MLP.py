from utils import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, MAX_LEN, LEARNING_RATE, set_seed
from preprocessing import prepare_data
import torch
import torch.nn as nn
import torch.optim as optim
from training_loop import train_model
import warnings 
warnings.filterwarnings('ignore')

set_seed()

train_path = './data/twitter_training.csv'
test_path = './data/twitter_validation.csv'
train_loader, test_loader, vocab, embedding_matrix = prepare_data(train_path, test_path)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=vocab["<pad>"]) # embedding layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Initialize model, loss function, and optimiser
model = MLP(EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=4)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2, factor=0.5)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# train the model
history = train_model(model, train_loader, test_loader, optimiser, criterion, DEVICE, epochs=EPOCHS)
