# IMDB Movie Review Sentiment Analysis

A simple web application for sentiment analysis of movie reviews from the IMDB dataset using a Recurrent Neural Network (RNN).

![image](https://github.com/iamsandeeprSand/IMDB_MovieReviewSentimentAnalysis/assets/139530620/87d9346f-67d8-493d-ad34-e1f0950f7625)


## Table of Contents

- [Demo](#demo)
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)

## Demo

You can find a live demo of this project [here](#).

## Introduction

This project is a simple web application for sentiment analysis of movie reviews from the IMDB dataset using a Recurrent Neural Network (RNN) built with TensorFlow and Streamlit. The model was trained on the IMDB dataset, and it predicts whether a given movie review expresses positive or negative sentiment.

## Features

- Predicts sentiment (positive/negative) of a movie review.
- Provides probability score along with the sentiment prediction.
- User-friendly interface built with Streamlit.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   ```

2. Change the directory:

   ```bash
   cd imdb-sentiment-analysis
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. After cloning the repository and installing the required dependencies, run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in your terminal to access the web application.
3. Enter a movie review in the text area provided and click on the "Predict" button.
4. The application will predict whether the review expresses positive or negative sentiment and provide a probability score.

## Technologies

- Python
- TensorFlow
- Streamlit
- NumPy
