import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

def extract_BoW(set_of_strings):
    
    vectorizer = CountVectorizer(stop_words='english', binary=False, ngram_range=(1, 1), token_pattern=r'(?u)\b\w\w+\b', min_df=1)
    x_train_df = pd.read_csv('data_reviews/x_train.csv')
    x_train = x_train_df['text']
    second_elements = [pair[1] for pair in set_of_strings]
    feature_vectors1 = vectorizer.fit_transform(x_train)
    feature_vectors2 = vectorizer.transform(second_elements)
    
    return feature_vectors2.toarray()