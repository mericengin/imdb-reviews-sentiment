import streamlit as st
import pickle
import sys
import os

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessing import preprocess_text

# Set base directory for consistent file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load Our Vectorizer and Models
@st.cache_resource
def load_resources():
    with open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'models', 'LogisticRegression_model.pkl'), "rb") as f:
        lr_model = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'models', 'NaiveBayes_model.pkl'), "rb") as f:
        nb_model = pickle.load(f)

    return vectorizer, lr_model, nb_model

vectorizer, lr_model, nb_model = load_resources()

# Streamlit app
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and choose a model to predict the sentiment!")

user_input = st.text_area("Movie Review", "")

model_choice = st.radio("Choose a model:", ("Logistic Regression", "Naive Bayes"))

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review!")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        model = lr_model if model_choice == 'Logistic Regression' else nb_model
        prediction = model.predict(vectorized_text)[0]
        
        if prediction == 'positive':
            label = "Positive 😊"
            st.markdown(f"<h3 style='color: green;'>Predicted Sentiment: {label}</h3>", unsafe_allow_html=True)
        else:
            label = "Negative 😠"
            st.markdown(f"<h3 style='color: red;'>Predicted Sentiment: {label}</h3>", unsafe_allow_html=True)