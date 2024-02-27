"""
This file provide a test of your feature extractor and classifier to make sure that they will work 
with the autograder. 

Please make sure that `model_loader` actually load your feature extractors and classifiers, NOT the 
provided stub objects. 
"""

import pandas as pd
import os
from model_loader1 import extract_BoW_features, classifier1, teamname  
from model_loader2 import extract_awesome_features, classifier2 
from model_loader2 import teamname as teamname_rep

def main():

    # Print the team name
    assert(isinstance(teamname, str))
    assert(teamname == teamname_rep)
    print("Team " + teamname + " is working on the project!")

    # overview the training data
    data_dir='data_reviews'
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test_6_samples.csv'))
    x_test_text = x_test_df.values.tolist()
    print("\nThe test inputs:")
    print(x_test_text)

    ## test the classifier for problem 1

    # this function should accept a list of N lists, with each internal list contains two strings
    # It should return a numpy array of size [N, F1]
    x_test1 = extract_BoW_features(x_test_text)
    assert(x_test1.shape[0] == len(x_test_text))

    # `classifier1` should be able to predict N probabilities from a feature matrix of size [N, F1]
    yhat_test1 = classifier1.predict_proba(x_test1)
    assert(yhat_test1.shape[0] == len(x_test_text))

    print("\nPredicted probabilities from the first model:")
    print(yhat_test1[:, 1])


    ## test the classifier for problem 2. 
    # NOTE: Please comment this block out if you don't need a test for the second classifier. 

    # this function should accept a list of N lists, with each internal list contains two strings
    # It should return a numpy array of size [N, F2]
    x_test2 = extract_awesome_features(x_test_text)
    assert(x_test2.shape[0] == len(x_test_text))

    # `classifier2` should be able to predict N probabilities from a feature matrix of size [N, F2]
    yhat_test2 = classifier2.predict_proba(x_test2)
    assert(yhat_test2.shape[0] == len(x_test_text))
    print("\nPredicted probabilities from the second model:")
    print(yhat_test2[:, 1])

    print("\nIf you see this line without errors or warnings, your feature extractor and classifier pass the test!")

    
if __name__ == '__main__':
    main()
