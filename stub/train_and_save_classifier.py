
"""Example code of saving classifier"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Just an example: training a classifier on random data
x_train = np.random.rand(100, 10)
y_train = (np.random.rand(100) > 0.5).astype(np.int32)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# save classifiers to pkl files
with open('classifier1.pkl','wb') as f:
    pickle.dump(classifier,f)

with open('classifier2.pkl','wb') as f:
    pickle.dump(classifier,f)



