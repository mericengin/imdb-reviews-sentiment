import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """This function preprocesses text (removing html tags, removing stopwords, lowercasing and tokenizing the texts)"""
    
    # Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)
    
    # Tokenize Text
    tokens = word_tokenize(text)
    
    # Lowercase Everything
    tokens = [word.lower() for word in tokens]
    
    # Remove Newlines
    tokens = [word.strip() for word in tokens]
    
    # Remove Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    return ' '.join(filtered_tokens)