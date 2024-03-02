import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import pickle

# Stub method for feature extractions. 
# You should implement your BoW and awesome feature extraction methods 
# in separate files and import them in `model_loader.py`


# def clean_text(text):
#     # Convert text to lowercase
#     # text = text.lower()
#     # Remove punctuation
#     #text = re.sub(r'[^\w\s]', '', text)
#     return text


def extract_BoW_features1(x_test_text):
    
    with open('stub/vectorizer1.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract just the text component from each [source, text] pair
    texts = [text[1] for text in x_test_text]
    
    # clean the text in the
    # Fit and transform the text data to a sparse matrix
    X = vectorizer.transform(texts).toarray()
    
    return X


def extract_BoW_features2(x_test_text):
    
    with open('stub/vectorizer2.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract just the text component from each [source, text] pair
    texts = [text[1] for text in x_test_text]
    
    # clean the text in the
    # Fit and transform the text data to a sparse matrix
    X = vectorizer.transform(texts).toarray()
    
    return X
















# def dumb_feature_extractor1(x_text):

#     x = np.random.random([len(x_text), 10])
    
#     return x

def dumb_feature_extractor2(x_text):

    x = np.ones([len(x_text), 10])
    
    return x




