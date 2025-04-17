import streamlit as st
import pickle
import sys
import os

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessing import preprocess_text
from visualization.confusion_matrix_plot import plot_conf_matrix
from visualization.classification_report_display import get_classification_report_df

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
        
    with open(os.path.join(BASE_DIR, 'data', 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, 'data', 'log_reg_preds.pkl'), 'rb') as f:
        log_reg_predictions = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, 'data', 'nb_preds.pkl'), 'rb') as f:
        naive_bayes_predictions = pickle.load(f)

    return vectorizer, lr_model, nb_model, y_test, log_reg_predictions, naive_bayes_predictions

vectorizer, lr_model, nb_model, y_test, log_reg_predictions, naive_bayes_predictions = load_resources()

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
        confidence = model.predict_proba(vectorized_text).max()
        
        if prediction == 'positive':
            st.markdown(f"<h3 style='color: green;'>Predicted Sentiment: {prediction.capitalize()} 😊</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>Predicted Sentiment: {prediction.capitalize()} 😠</h3>", unsafe_allow_html=True)
        
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Metrics Section (Tabs)
        with st.expander("🔍 Model Performance Overview"):
            tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📋 Classification Report"])
            
            with tab1:
                st.subheader("Confusion Matrix")
                fig = plot_conf_matrix(y_test, log_reg_predictions if model_choice == "Logistic Regression" else naive_bayes_predictions)
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Classification Report")
                report = get_classification_report_df(y_test, log_reg_predictions if model_choice == "Logistic Regression" else naive_bayes_predictions)
                st.text(report)  # Display the classification report