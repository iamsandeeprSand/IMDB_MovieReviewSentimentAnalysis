import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

# Parameters
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words (among top max_features most common words)

# Load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# Preprocess the data
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Load the pre-trained RNN model
model_rnn = load_model("imdb_rnn_model.h5")

# Evaluate the pre-trained model
loss, accuracy = model_rnn.evaluate(test_data, test_labels)
st.write("Test Accuracy:", accuracy)

# Make predictions
def predict_sentiment(text):
    word_to_index = imdb.get_word_index()
    text = text.split()
    text = [word_to_index[word] if word in word_to_index and word_to_index[word] < max_features else 0 for word in text]
    text = pad_sequences([text], maxlen=maxlen)
    prediction = model_rnn.predict(text)
    return prediction[0][0]

st.title("IMDB Movie Review Sentiment Analysis")
review = st.text_area("Enter a movie review:")
if st.button("Predict"):
    if review:
        prediction = predict_sentiment(review)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.write("Sentiment:", sentiment, "| Probability:", prediction)
    else:
        st.warning("Please enter a movie review.")
