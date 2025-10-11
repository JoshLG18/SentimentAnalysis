# training_loop.py
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from utils import EPOCHS
from tqdm import tqdm


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------
def train_one_epoch(model, dataloader, optimiser, scheduler, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc="Training")

    for batch_idx, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at batch {batch_idx}, skipping update")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# --------------------------------------------------
# Improved evaluation (with per-class metrics)
# --------------------------------------------------
def evaluate(model, dataloader, criterion, device, label_names=None, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Weighted and macro averages
    acc = accuracy_score(all_labels, all_preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # Per-class detailed report
    report = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True
    )

    results = {
        "loss": total_loss / len(dataloader),
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "report": report,
    }

    if return_preds:
        results["y_true"] = all_labels
        results["y_pred"] = all_preds

    return results


# --------------------------------------------------
# Training loop with early stopping
# --------------------------------------------------
def train_model(save_path, model, train_loader, val_loader, optimiser, scheduler,
                criterion, device, epochs=EPOCHS, patience=3, label_names=None):

    history = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision_weighted": [], "recall_weighted": [], "f1_weighted": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimiser, scheduler, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device, label_names)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1_w: {metrics['f1_weighted']:.4f}")

        # Store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["accuracy"].append(metrics["accuracy"])
        history["precision_weighted"].append(metrics["precision_weighted"])
        history["recall_weighted"].append(metrics["recall_weighted"])
        history["f1_weighted"].append(metrics["f1_weighted"])

        # Save best model
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_metrics = metrics.copy()
            best_metrics["train_loss"] = train_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"✅ New best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered")
                break

    model.load_state_dict(torch.load(save_path))
    return history, best_metrics
