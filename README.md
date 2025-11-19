# 🧠 Comparative Evaluation of Neural Architectures for Sentiment Analysis on Financial News: From MLPs to Transformers

## Project Overview

This project addresses the growing interest in integrating **sentiment analysis** into traditional financial models to account for the emotional and psychological factors that influence investor decisions. This study investigates whether increasing architectural complexity in deep learning models (MLP, LSTM, Transformer) leads to performance gains when classifying sentiment in **financial news headlines**.

Traditional machine learning (ML) methods often underperform in complex domains like finance due to the need for domain-specific feature engineering, limiting their ability to understand subtle sentiments. To combat this, deep learning methods—such as RNNs, LSTMs, and Transformers—offer a more accurate alternative by capturing sequential patterns.

***

## Methodology and Models

Sentiment prediction was treated as a **three-class classification problem** (positive, negative, or neutral). All neural models were trained on **frozen FinBERT embeddings**. FinBERT, a finance-specific language model, was used because its domain-specific training typically outperforms general-purpose models in financial text tasks.

### Models Evaluated

| Architecture | Description | Purpose |
| :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF embeddings + Logistic Regression. | Traditional ML baseline. |
| **MLP** | Three fully connected layers ($\mathbf{768 \rightarrow 256 \rightarrow 128}$) with ReLU activation, dropout (0.3), and Layer Norm. | Neural baseline architecture. |
| **LSTM** | Three-layer **bidirectional LSTM** (hidden size $=256$) with dropout (0.3), Layer Norm, and an attention layer. | Designed to model sequential relationships. |
| **Transformer** | Four-layer Transformer Encoder with 8 attention heads, Layer Norm, and a two-layer MLP classifier. | State-of-the-art architecture utilizing self-attention. |

### Dataset & Preprocessing

The project used the **English Financial News dataset** ($\approx 27,000$ headlines). The dataset has a class imbalance, with **neutral sentiment** being the majority class ($62.5\%$). This imbalance was addressed during training through stratified sampling and weighted metrics.

Preprocessing included:
* Removal of URLs, stock tickers (e.g., \$AAPL), special symbols, and unnecessary whitespace.
* Tokenisation using the `ProsusAI/finbert` tokenizer.
* Truncating input sequences to a maximum length of 64 tokens.

***

## Key Results and Findings

Results indicate that all FinBERT-based neural architectures **significantly outperformed** the TF-IDF + Logistic Regression baseline.

| Metric | Logistic Regression | MLP | LSTM | Transformer |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | $0.854$ | $0.882$ | $0.897$ | **$0.903$** |
| **F1 (Macro)** | $0.814$ | $0.853$ | $0.871$ | **$0.876$** |
| **ROC-AUC (Macro)** | $0.942$ | $0.962$ | $0.966$ | **$0.969$** |
| **Training Time (min)** | $0.057$ | $32.578$ | $32.602$ | $35.226$ |

### Conclusions and Trade-Offs

1.  **Complexity vs. Performance:** Increasing architectural complexity led to classification improvements, although with diminishing returns. The Transformer achieved the highest accuracy; however, the gains over simpler FinBERT-based neural models were small.
2.  **Embeddings over Architecture:** The results suggest that **high-quality embeddings may matter more** than the classifier's architectural complexity.
3.  **Computational Cost:** Neural architectures required a $570$x to $620$x increase in training time compared to the Logistic Regression baseline for up to a $4.9\%$ absolute accuracy gain.
4.  **Limitations:** All models struggled with **negative sentiment** and **implicit positive sentiment** (where headlines lacked clear growth indicators), often misclassifying them as neutral.

***

## Skills Used

| Category | Skills |
| :--- | :--- |
| **Natural Language Processing (NLP)** | Sentiment Analysis, FinBERT Embeddings, Tokenization, TF-IDF. |
| **Deep Learning** | Transformer Encoder, Bidirectional LSTM, Multi-Layer Perceptron (MLP), Self-Attention, AdamW Optimizer, Cross-Entropy Loss, Gradient Clipping. |
| **Research & Evaluation** | Comparative Modeling, Performance Benchmarking, F1/ROC-AUC Metrics, Qualitative Analysis, Architectural Trade-Offs. |

***

**Next Steps & Future Work:**

* Address the class imbalance through data augmentation.
* Evaluate whether the sentiment predictions can inform market movements more accurately than traditional quantitative models.
* Explore more extensive model optimization and statistical significance testing.
