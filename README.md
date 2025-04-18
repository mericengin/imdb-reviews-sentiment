# 🎬 IMDB Movie Reviews Sentiment Analysis

This project performs sentiment analysis on IMDB reviews using machine learning techniques such as **Logistic Regression** and **Multinomial Naive Bayes**. It classifies reviews as either **positive** or **negative** based on the content of the review using a trained TF-IDF vectorizer and is deployed as a web app using Streamlit.

## 🚀 Live Demo

🔗 [Click here to try the app](https://imdb-reviews-sentiment-zsqdj8xeqfysdjptmrmopg.streamlit.app)

## 🧠 Models Used

- Logistic Regression
- Multinomial Naive Bayes

## 🧰 Tech Stack

- Python
- Scikit-learn
- NLTK
- Pandas, NumPy
- Streamlit
- Matplotlib & Seaborn (for visualizations)

## 📦 Features

- Text Preprocessing (stopword removal, tokenization)
- TF-IDF Vectorization
- Word Clouds for Positive and Negative Reviews
- Confusion Matrix and Classification Report Visualization
- Prediction Confidence Score
- Sentiment Distribution Chart

## 📁 Project Structure

- **`app/main.py`**: Streamlit app that provides the user interface.
- **`models/`**: Folder containing saved models and vectorizers.
    - `tfidf_vectorizer.pkl`
    - `LogisticRegression_model.pkl`
    - `NaiveBayes_model.pkl`
- **`notebooks/`**: Folder containing Jupyter notebooks for EDA, training, evaluation.
- **`data/`**: Contains raw and processed data.
    - **`raw/`**: Raw IMDB data before preprocessing.
    - **`processed/`**: Cleaned and vectorized data.
- **`nltk_data/`**: Local NLTK downloads folder for deployment compatibility.
- **`requirements.txt`**: Python dependencies.
- **`.gitignore`**: File patterns to be ignored by Git.
- **`README.md`**: This file.

## Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis