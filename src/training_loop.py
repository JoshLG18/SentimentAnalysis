# training_loop.py
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
from utils import EPOCHS
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimiser, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Training")
    for batch_idx, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimiser.step()
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


def train_model(save_path,model, train_loader, val_loader, optimizer, criterion, device, epochs=EPOCHS, patience=3):
    history = {"train_loss": [], "val_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None  # to store metrics of best model

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)

        # Logging
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["accuracy"].append(metrics["accuracy"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])

        # Check for improvement
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_metrics = metrics.copy()   # save metrics of the best model
            best_metrics["train_loss"] = train_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"✅ New best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered")
                break
    
    # Load best model before returning
    model.load_state_dict(torch.load(save_path))
    return history, best_metrics

