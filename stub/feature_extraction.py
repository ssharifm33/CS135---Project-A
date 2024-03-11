import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import pickle

# Stub method for feature extractions. 
# You should implement your BoW and awesome feature extraction methods 
# in separate files and import them in `model_loader.py`




def extract_BoW_features1(x_test_text):
    
    with open('vectorizer1.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract just the text component from each [source, text] pair
    texts = [text[1] for text in x_test_text]
    
    # clean the text in the
    # Fit and transform the text data to a sparse matrix
    X = vectorizer.transform(texts).toarray()
    
    return X


def extract_BoW_features2(x_test_text):
    
    with open('vectorizer2.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract just the text component from each [source, text] pair
    texts = [text[1] for text in x_test_text]
    
    # clean the text in the
    # Fit and transform the text data to a sparse matrix
    X = vectorizer.transform(texts).toarray()
    
    return X


