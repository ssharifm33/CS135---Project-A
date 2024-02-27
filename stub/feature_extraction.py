import numpy as np

# Stub method for feature extractions. 
# You should implement your BoW and awesome feature extraction methods 
# in separate files and import them in `model_loader.py`

def dumb_feature_extractor1(x_text):

    x = np.random.random([len(x_text), 10])
    
    return x

def dumb_feature_extractor2(x_text):

    x = np.ones([len(x_text), 10])
    
    return x




