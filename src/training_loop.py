# import the libraries
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from utils import EPOCHS
from tqdm import tqdm

# Train one epoch
def train_one_epoch(model, dataloader, optimiser, scheduler, criterion, device):
    model.train() # set the model to training mode
    total_loss = 0 # initiatlise total loss to 0

    # set up the loop for visual progress bar
    loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc="Training")

    for batch_idx, batch in loop: # set up the loop for mini-batching
        # inidialise the input data, attention amsk and labels
        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimiser.zero_grad() # set the gradients of the optimsier back to 0

        # compute the outputs of the prediction
        outputs = model(input_ids=input_ids, attention_mask=attention_mask) 

        loss = criterion(outputs, labels) # compute loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at batch {batch_idx}, skipping update")
            continue

        loss.backward() # backpropagation - compute the gradients

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step() # update the weights
        #add the loss from this batch
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item()) # add the current loss to the end of the progress bar

    return total_loss / len(dataloader) # return total loss / number of batches


# Evaluation
def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval() # set the evaluation mode
    total_loss = 0.0 # initialse the total loss to 0
    all_preds, all_labels = [], [] # create empty lists for the preds and labels
    test_texts = []
    test_labels = []  

    with torch.no_grad(): 
        for batch in dataloader: # mini-batching
            # set up the input, attention and labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # compute the output, loss and increment the total loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # prediction chooses the largest of the output logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy()) # add the pred to the list
            all_labels.extend(labels.cpu().numpy()) # add the label to the list

            decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            test_texts.extend(decoded_texts)
            test_labels.extend(labels.cpu().numpy())


    # convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute the Metrics 
    acc = accuracy_score(all_labels, all_preds) # accuracy

    # precision, recall and f1; weighted and overall/macro
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    # create the classification report which gives the metrics for each class
    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True)

    #  create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # TO-DO Add in AUC Curves

    label_names = ["negative", "neutral", "positive"]
    example_tracker = {"correct": {}, "wrong": {}}

    for text, true, pred in zip(test_texts, test_labels, all_preds):
        true_name = label_names[true] if true < len(label_names) else str(true)
        pred_name = label_names[pred] if pred < len(label_names) else str(pred)

        if true == pred and true_name not in example_tracker["correct"]:
            example_tracker["correct"][true_name] = {
                "text": text,
                "true": true_name,
                "pred": pred_name,
            }
        if true != pred and true_name not in example_tracker["wrong"]:
            example_tracker["wrong"][true_name] = {
                "text": text,
                "true": true_name,
                "pred": pred_name,
            }

    # Store results
    results = {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "examples": example_tracker
    }

    return results

# Training loop with early stopping
def train_model(save_path, model, train_loader, val_loader, optimiser, scheduler,
                criterion, device,tokeniser, epochs=EPOCHS, patience=5, ):
    
    # set up the history dictionary to store the training metrics
    history = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision_weighted": [], "recall_weighted": [], "f1_weighted": []
    }

    best_val_loss = float("inf") # initialse best val loss
    patience_counter = 0 # initialise patience tracker
    best_metrics = None # initalise best metrics to 0

    for epoch in range(epochs): # loops through the number of epochs

        # train the model
        train_loss = train_one_epoch(model, train_loader, optimiser, scheduler, criterion, device)
        
        # evaluate the model
        metrics = evaluate(model, val_loader, criterion, device, tokeniser)

        # print the metrics - allows tracking training progress
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

        # monitor loss and take a step if needed
        scheduler.step(metrics['loss'])

        # Save best model and track patience
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_metrics = metrics.copy()
            best_metrics["train_loss"] = train_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"✅ New best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1 # if the better model isn't found increaase patience counter
            if patience_counter >= patience: # if patients is reached early stop
                print("⏹ Early stopping triggered")
                break

    model.load_state_dict(torch.load(save_path)) # save the best model
    return history, best_metrics # return the history and the best metrics
