# Sentiment Analysis Project

This project performs sentiment analysis on IMDB reviews using machine learning techniques such as Logistic Regression and Multinomial Naive Bayes.

## Project Overview

The goal of this project is to classify IMDB reviews as either **positive** or **negative** based on the review text. The models are trained using the `scikit-learn` library, and various preprocessing steps such as tokenization, lemmatization, and TF-IDF vectorization are applied to the text data.

## Project Structure

- **`main.py`**: The main script that runs the sentiment analysis process (To be Written As a Streamlit App).
- **`notebooks/`**: Folder containing Jupyter notebooks for exploration and analysis.
- **`models/`**: Folder for saving trained machine learning models (to be added later).
- **`data/`**: Contains raw and processed data for training and testing.
    - **`raw/`**: Raw data (IMDB reviews) before preprocessing.
    - **`processed/`**: Processed data after cleaning and feature extraction.
- **`README.md`**: This file, describing the project.
- **`.gitignore`**: Configuration to ignore unnecessary files from Git version control.

## Setup & Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
