# [cite_start]Comparative Evaluation of Neural Architectures for Sentiment Analysis on Financial News: From MLPs to Transformers [cite: 1]

## Project Overview

[cite_start]This project addresses the growing interest in integrating sentiment analysis into traditional financial models to account for the emotional and psychological factors that influence investor decisions[cite: 7]. [cite_start]While traditional Machine Learning (ML) methods provide a foundation for sentiment analysis, they may underperform in complex financial domains due to the need for domain-specific feature engineering[cite: 9].

[cite_start]This study investigates whether increasing architectural complexity in deep learning models (MLP, LSTM, Transformer) leads to performance gains when classifying sentiment in financial news headlines[cite: 16].

## Methodology and Models

[cite_start]Sentiment prediction was treated as a three-class classification problem (positive, negative, or neutral)[cite: 29]. [cite_start]All neural models used pre-trained **FinBERT embeddings** (extracted as 768-dimensional [CLS] embeddings) from Prosus AI[cite: 30, 32, 152]. [cite_start]FinBERT was selected because its domain-specific training typically outperforms general-purpose models in financial text tasks[cite: 33].

### Models Evaluated

| Architecture | Description | Purpose |
| :--- | :--- | :--- |
| **Logistic Regression** | [cite_start]TF-IDF embeddings + Logistic Regression[cite: 30, 31]. | [cite_start]Served as a traditional ML baseline for comparison against FinBERT-based models[cite: 31]. |
| **MLP** | [cite_start]Three fully connected layers ($768\rightarrow256\rightarrow128$) with ReLU activation, dropout (0.3), and Layer Norm[cite: 152]. | [cite_start]Neural baseline architecture[cite: 34]. |
| **LSTM** | [cite_start]Three-layer **bidirectional LSTM** (hidden size $=256$) with dropout (0.3), Layer Norm, and an attention layer[cite: 152]. | [cite_start]Designed to model sequential relationships in the text[cite: 34]. |
| **Transformer** | [cite_start]Four-layer Transformer Encoder with 8 attention heads and a two-layer MLP classifier[cite: 152]. | [cite_start]State-of-the-art architecture utilizing self-attention[cite: 34]. |

### Dataset & Preprocessing

[cite_start]The project used the **English Financial News dataset** [cite: 19, 126][cite_start], consisting of $\approx 27,000$ headlines labeled as positive, negative, or neutral[cite: 19, 20]. [cite_start]The dataset has a class imbalance, with neutral sentiment being the majority class ($62.5\%$)[cite: 26, 50].

[cite_start]Preprocessing included lowercasing the text, removal of URLs, stock tickers (e.g., \$AAPL), special symbols, and unnecessary whitespace[cite: 152].

## Key Results and Findings

[cite_start]Results indicate that all FinBERT-based neural architectures **significantly outperformed** the TF-IDF + Logistic Regression baseline, highlighting the importance of high-quality embeddings[cite: 38, 59].

| Metric | Logistic Regression | MLP | LSTM | Transformer |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | [cite_start]$0.854$ [cite: 40] | [cite_start]$0.882$ [cite: 40] | [cite_start]$0.897$ [cite: 40] | [cite_start]**$0.903$** [cite: 40] |
| **F1 (Macro)** | [cite_start]$0.814$ [cite: 41] | [cite_start]$0.853$ [cite: 41] | [cite_start]$0.871$ [cite: 41] | [cite_start]**$0.876$** [cite: 41] |
| **ROC-AUC (Macro)** | [cite_start]$0.942$ [cite: 41] | [cite_start]$0.962$ [cite: 41] | [cite_start]$0.966$ [cite: 41] | [cite_start]**$0.969$** [cite: 41] |
| **Training Time (min)** | [cite_start]$0.057$ [cite: 42] | [cite_start]$32.578$ [cite: 42] | [cite_start]$32.602$ [cite: 42] | [cite_start]$35.226$ [cite: 42] |

### Conclusions and Trade-Offs

1.  [cite_start]**Complexity vs. Performance:** Increasing architectural complexity led to classification improvements, although with diminishing returns[cite: 39, 46]. [cite_start]The Transformer achieved the highest accuracy ($90.3\%$), followed closely by the LSTM ($89.7\%$)[cite: 40].
2.  [cite_start]**Embeddings over Architecture:** The small gains between FinBERT-based models (MLP to Transformer: $2.1\%$) suggest that **high-quality embeddings matter more** than the classifier's architectural complexity[cite: 47, 59].
3.  [cite_start]**Computational Cost:** Neural architectures required a $570$x to $620$x increase in training time compared to the Logistic Regression baseline for an absolute accuracy gain of up to $4.9\%$[cite: 42, 113].
4.  [cite_start]**Limitations:** All models performed best on the neutral class [cite: 106] [cite_start]due to the dataset's skew towards neutral sentiment[cite: 50]. [cite_start]Furthermore, models consistently struggled with **implicit positive sentiment** (headlines describing company strengths rather than clear growth indicators)[cite: 44, 52].

## Skills Used

| Category | Skills |
| :--- | :--- |
| **Natural Language Processing (NLP)** | Sentiment Analysis, FinBERT Embeddings, Tokenization, TF-IDF. |
| **Deep Learning** | [cite_start]Transformer Encoder, Bidirectional LSTM, Multi-Layer Perceptron (MLP), Self-Attention, AdamW Optimizer, Cross-Entropy Loss, Gradient Clipping[cite: 34, 152]. |
| **Research & Evaluation** | [cite_start]Comparative Modeling, Performance Benchmarking, F1/ROC-AUC Metrics, Qualitative Analysis, Statistical Significance[cite: 41, 60]. |

---

**Next Steps & Future Work:**

* [cite_start]Address the class imbalance using techniques like data augmentation[cite: 61].
* [cite_start]Evaluate whether the small performance gains from FinBERT-based models justify the increased training cost[cite: 61].
* [cite_start]Explore the practical application of these sentiment predictions to inform market movements[cite: 63].
* [cite_start]Discuss the need for SHAP or LIME analyses on neural models for improved interpretability in high-stakes financial applications[cite: 55].

[cite_start][View the full presentation here] (https://youtu.be/4NpsymyU5d8) [cite: 140]