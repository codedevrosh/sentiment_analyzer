import streamlit as st
import tensorflow as tf
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first run)
nltk.download('stopwords')

# App title
st.set_page_config(page_title="AI Product Review Analyzer", page_icon="🤖")

st.title(" AI Product Review Analyzer")
st.write("Analyze product reviews and detect whether the sentiment is **Positive, Neutral, or Negative**.")

# Load model with caching (faster performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.h5")

model = load_model()

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return pickle.load(open("tokenizer.pkl", "rb"))

tokenizer = load_tokenizer()

max_length = 100

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Sentiment prediction function
def predict_sentiment(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)

    pred = model.predict(padded)[0]

    label = np.argmax(pred)

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    confidence = np.max(pred)

    return labels[label], confidence


# Input box
review = st.text_area("Enter your product review:")

# Button
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing review..."):
            sentiment, confidence = predict_sentiment(review)

        st.success(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: **{confidence:.2f}**")


st.markdown("---")
st.caption("Built with TensorFlow, NLP, and Streamlit")