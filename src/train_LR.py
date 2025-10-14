# import all libraries
import warnings
import time
import json
import joblib
from utils import set_seed, save_metrics_and_history
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

warnings.filterwarnings("ignore")
set_seed()

# Prepare data
def prepare_data_tfidf(test_size=0.2, random_state=123):
    # Load raw dataset (same as your FinBERT setup)
    raw_dataset = load_dataset("lukecarlate/english_finance_news")
    df = raw_dataset["train"].to_pandas()[["newscontents", "label"]]
    df.columns = ["text", "label"]

    # Train/test split (stratified)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    train_texts, test_texts = train_df["text"].tolist(), test_df["text"].tolist()
    train_labels, test_labels = train_df["label"].tolist(), test_df["label"].tolist()

    return train_texts, train_labels, test_texts, test_labels


# Train and evaluate model
def train_baseline_tfidf_lr(save_path="./results/saved_models/tfidf_lr.pkl"):
    print("Preparing data...")
    train_texts, train_labels, test_texts, test_labels = prepare_data_tfidf()

    # Define TF-IDF + Logistic Regression pipeline
    print("Initialising TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    model = LogisticRegression(class_weight='balanced')

    # Train model
    print("Training baseline model...")
    start_time = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    model.fit(X_train, train_labels)
    training_time = time.time() - start_time
    print(f"✅ Training complete in {training_time:.2f} seconds.")

    # Predictions
    preds = model.predict(X_test)

    # Metrics and examples
    acc = accuracy_score(test_labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(test_labels, preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
    report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_labels, preds)

    label_names = ["negative", "neutral", "positive"]
    example_tracker = {"correct": {}, "wrong": {}}

    for text, true, pred in zip(test_texts, test_labels, preds):
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
    best_metrics = {
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

    # Save model and vectorizer
    joblib.dump({"model": model, "vectorizer": vectorizer}, save_path)
    print(f"💾 Model saved to {save_path}")

    # Save metrics using your project utility
    save_metrics_and_history(best_metrics, history=None, training_time=training_time)

    print("✅ Baseline complete.")
    print(f"Accuracy: {acc:.4f} | F1_w: {f1_w:.4f} | F1_m: {f1_m:.4f}")

    return best_metrics


# Run baseline
if __name__ == "__main__":
    train_baseline_tfidf_lr()
