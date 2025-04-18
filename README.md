# 🎬 IMDB Movie Reviews Sentiment Analysis

This project performs sentiment analysis on IMDB reviews using machine learning techniques such as **Logistic Regression** and **Multinomial Naive Bayes**. It classifies reviews as either **positive** or **negative** based on the content of the review using a trained TF-IDF vectorizer and is deployed as a web app using Streamlit.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://imdb-reviews-sentiment-zsqdj8xeqfysdjptmrmopg.streamlit.app)

---

## 🧠 Models Used

- Logistic Regression
- Multinomial Naive Bayes

---

## 🧰 Tech Stack

- Python
- Scikit-learn
- NLTK
- Pandas, NumPy
- Streamlit
- Matplotlib & Seaborn (for visualizations)

---

## 📦 Features

- Text Preprocessing (stopword removal, tokenization)
- TF-IDF Vectorization
- Word Clouds for Positive and Negative Reviews
- Confusion Matrix and Classification Report Visualization
- Prediction Confidence Score
- Sentiment Distribution Chart

---

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

2. **Create and activate a virtual environment (recommended)**:
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Create and activate a virtual environment (recommended)**:
   pip install -r requirements.txt

## NLTK Resources

Make sure the following NLTK resources are available. These are required for tokenization and stopword removal.

You can download them programmatically:

   import nltk
   import os

   nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
   nltk.data.path.append(nltk_data_path)

   try:
    nltk.data.find('tokenizers/punkt')
   except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

   try:
    nltk.data.find('corpora/stopwords')
   except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

Or manually via Python console:

   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

## Run the App

Make sure your models are placed in the models/ directory.

Then launch the Streamlit app:
   streamlit run app/main.py