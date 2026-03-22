# AI Product Review Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) project that analyzes product reviews and classifies them into **Negative, Neutral, or Positive sentiments** using a Deep Learning **LSTM model**.

This project covers the full machine learning workflow including **data preprocessing, model training, evaluation, and deployment using Streamlit** to build an interactive web application.

---

# Project Overview

Understanding customer sentiment helps businesses evaluate product performance and customer satisfaction.  

This project builds a sentiment analysis system that automatically classifies product reviews into sentiment categories.

The project workflow includes:

- Text preprocessing and cleaning
- Tokenization and sequence conversion
- Deep learning model training using LSTM
- Model evaluation
- Deployment with Streamlit

The final application allows users to enter a product review and receive sentiment predictions instantly.

---

# Features

- Text preprocessing using NLTK
- Tokenization and padding for sequence modeling
- Deep learning sentiment classification using LSTM
- Multi-class sentiment prediction
- Interactive Streamlit web application
- Model and tokenizer persistence for deployment

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning model |
| NLTK | Text preprocessing |
| NumPy | Numerical computation |
| Streamlit | Web application |
| Pickle | Saving tokenizer |

---

# Dataset

The dataset contains product reviews labeled with three sentiment categories:

- Negative
- Neutral
- Positive

Dataset distribution:

| Sentiment | Count |
|---|---|
| Negative | 48,215 |
| Positive | 30,382 |
| Neutral | 6,995 |

The dataset shows class imbalance, particularly for Neutral reviews.

---

# Model Architecture

The sentiment classifier is built using an LSTM neural network designed for sequence-based text data.

Model structure:

Embedding Layer  
LSTM Layer  
Dense Layer (ReLU)  
Softmax Output Layer  

### Layer Description

Embedding Layer  
Converts words into dense vector representations.

LSTM Layer  
Captures contextual relationships between words in the sequence.

Dense Layer  
Learns higher-level patterns from extracted features.

Softmax Output Layer  
Predicts probabilities for each sentiment class.

---

# Model Performance

The trained model achieved:

Accuracy: **79.0**

This means the model correctly classifies approximately **79 out of 100 reviews**.

---

# Project Workflow

## 1. Data Preprocessing

Text cleaning includes:

- Converting text to lowercase
- Removing URLs
- Removing special characters
- Removing stopwords

---

## 2. Tokenization

Reviews are converted into numerical sequences using Keras Tokenizer.

---

## 3. Sequence Padding

All sequences are padded to a fixed length of **100 tokens**.

This ensures consistent input size for the neural network.

---

## 4. Model Training

The LSTM model learns patterns from text data to predict sentiment categories.

Training parameters:

- Epochs: 10  
- Batch Size: 128  
- Loss Function: Sparse Categorical Crossentropy  
- Optimizer: Adam  

---

## 5. Model Saving

After training, the model and tokenizer are saved. This allows the model to be reused without retraining.

---

# Streamlit Web Application

A Streamlit interface allows users to interact with the trained model.

Users can:

1. Enter a product review
2. Click Analyze
3. Receive predicted sentiment instantly

The application will open in your browser.

---

# Project Structure
```
sentiment-analysis
│
├── sentiment_model.keras
├── tokenizer.pkl
├── app.py
├── sentiment_analysis.py
├── requirements.txt
└── README.md
```

# Conclusion

This project demonstrates how Natural Language Processing and deep learning can be applied to analyze customer sentiment in product reviews. By preprocessing textual data, converting it into numerical sequences, and training an LSTM-based neural network, the model is able to classify reviews into Negative, Neutral, and Positive sentiments.

The model achieved an accuracy of approximately **79.0%**, indicating strong performance in identifying sentiment patterns within review text. The project also highlights the importance of text cleaning, tokenization, sequence padding, and proper model architecture when working with NLP tasks.

An interactive Streamlit application was developed to allow users to input product reviews and receive real-time sentiment predictions. This makes the model practical and easy to use in real-world scenarios such as customer feedback analysis, product evaluation, and brand monitoring.

Overall, the project showcases a complete end-to-end workflow for building, training, evaluating, and deploying a sentiment analysis system using deep learning techniques.
