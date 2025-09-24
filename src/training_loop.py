# training_loop.py
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=None):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Epoch {epoch}")
    for batch_idx, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # update progress bar with current loss
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)



def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()
            
            # Store for metrics
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=5):
    history = {"train_loss": [], "test_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1)

        # Evaluate
        metrics = evaluate(model, test_loader, criterion, device)

        # Save to history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(metrics["loss"])
        history["accuracy"].append(metrics["accuracy"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])

        # Print live after each epoch
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
    
    return history
