import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from feature_extraction import extract_BoW_features1
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC


def train_count_vectorizer(x_train):
    texts = [text[1] for text in x_train]
    
    # clean text here
    # add stop words
    
    vectorizer = CountVectorizer()
    
    vectorizer.fit_transform(texts)
    
    return vectorizer

def train_other_vectorizer(x_train):
    texts = [text[1] for text in x_train]
    
    # clean text here
    # add stop words
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    vectorizer.fit_transform(texts)
    
    return vectorizer



# Load your training data
x_train_df = pd.read_csv('/Users/sammiller/Desktop/CS135/projA-release/data_reviews/x_train.csv')
y_train_df = pd.read_csv('/Users/sammiller/Desktop/CS135/projA-release/data_reviews/y_train.csv')
y_train = y_train_df['is_positive_sentiment']

# y_train = np.array(y_train_df['is_positive_sentiment'])
x_train = x_train_df.values.tolist()

# clean the text for

# run grid search


vectorizer = train_count_vectorizer(x_train)
vectorizer_Tfid = train_other_vectorizer(x_train)

with open('/Users/sammiller/Desktop/CS135/projA-release/stub/vectorizer1.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('/Users/sammiller/Desktop/CS135/projA-release/stub/vectorizer2.pkl', 'wb') as f:
    pickle.dump(vectorizer_Tfid, f)

X_train = extract_BoW_features1(x_train)


# Initialize and train your LogisticRegression classifier
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter if needed

classifier2 = SVC(kernel='rbf', gamma='scale', probability=True)


classifier.fit(X_train, y_train)
classifier2.fit(X_train, y_train)


# Save classifiers to pkl files
with open('/Users/sammiller/Desktop/CS135/projA-release/stub/classifier1.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('/Users/sammiller/Desktop/CS135/projA-release/stub/classifier2.pkl', 'wb') as f:
    pickle.dump(classifier2, f)



