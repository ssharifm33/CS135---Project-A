import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Assuming these are your custom feature extraction methods,
# imported from their respective modules.
from feature_extraction import extract_BoW_features1, extract_BoW_features2

def train_count_vectorizer(x_train):
    texts = [text[1] for text in x_train]
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(texts)
    return vectorizer

def train_other_vectorizer(x_train):
    texts = [text[1] for text in x_train]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    vectorizer.fit_transform(texts)
    return vectorizer

# Load your training data
x_train_df = pd.read_csv('/Users/sammiller/Desktop/CS135/projA-release/data_reviews/x_train.csv')
y_train_df = pd.read_csv('/Users/sammiller/Desktop/CS135/projA-release/data_reviews/y_train.csv')
y_train = y_train_df['is_positive_sentiment']
x_train = x_train_df.values.tolist()

# Train vectorizers
vectorizer = train_count_vectorizer(x_train)
vectorizer_Tfid = train_other_vectorizer(x_train)

# Save vectorizers
with open('vectorizer1.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('vectorizer2.pkl', 'wb') as f:
    pickle.dump(vectorizer_Tfid, f)

# Feature extraction for each model
X_train1 = extract_BoW_features1(x_train)
X_train2 = extract_BoW_features2(x_train)

# Define the parameter grid for LogisticRegression
param_grid_lr = {
    'C': np.logspace(-7, 7, 20),
    'penalty': ['l1', 'l2'],
}

# Initialize the GridSearchCV object for LogisticRegression
#grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear'), param_grid_lr, cv=5, scoring='roc_auc', verbose=1)
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear'), 
                              param_grid_lr, cv=5, scoring='roc_auc', verbose=1, return_train_score=True)

# Fit GridSearchCV for LogisticRegression
grid_search_lr.fit(X_train1, y_train)

# Define the parameter grid for MultinomialNB
param_grid_nb = {
    'alpha': np.linspace(0.1, 2, 20),
}

# Initialize the GridSearchCV object for MultinomialNB
grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, scoring='roc_auc', verbose=1, return_train_score=True)

# Fit GridSearchCV for MultinomialNB
grid_search_nb.fit(X_train2, y_train)

# Save the best estimator
best_nb = grid_search_nb.best_estimator_

# Initialize StratifiedKFold for reproducibility
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


# Access the best set of parameters and retrain the classifiers
best_lr = LogisticRegression(**grid_search_lr.best_params_)


best_lr.fit(X_train1, y_train)

# # We will analyze one fold for false positives and negatives
# for train_index, test_index in skf.split(X_train2, y_train):
#     # Split data into training and test sets for this fold
#     X_train_fold, X_test_fold = X_train2[train_index], X_train2[test_index]
#     y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
#     original_test_sentences = np.array(x_train)[test_index]  # Retrieve original sentences for the test set
    
#     # Train the classifier using best parameters from the grid search
#     best_nb.fit(X_train_fold, y_train_fold)
    
#     # Make predictions on the test fold
#     predictions = best_nb.predict(X_test_fold)
    
#     # Determine indices for false positives and negatives
#     false_positives_indices = test_index[(predictions == 1) & (y_test_fold == 0)]
#     false_negatives_indices = test_index[(predictions == 0) & (y_test_fold == 1)]
    
#     # Retrieve original sentences based on false positive and negative indices
#     false_positives_sentences = original_test_sentences[(predictions == 1) & (y_test_fold == 0)]
#     false_negatives_sentences = original_test_sentences[(predictions == 0) & (y_test_fold == 1)]

#     # Display some representative false positive and negative examples
#     print("Some representative false positive examples for MultinomialNB:")
#     for i, text in enumerate(false_positives_sentences[:5]):  # Show just the first 5 examples
#         print(f"Example {i+1}: {text[1]}")  # Assuming text is a list of lists where the actual text is in the second position

#     print("\nSome representative false negative examples for MultinomialNB:")
#     for i, text in enumerate(false_negatives_sentences[:5]):  # Show just the first 5 examples
#         print(f"Example {i+1}: {text[1]}")
    
#     # Break after analyzing the first fold
#     break



# TODO: Add code here to display false positives and negatives
# This may involve transforming the feature vectors back to text
# and manually reviewing them or summarizing their properties.

# Output the best hyperparameters
print("Best hyperparameters for MultinomialNB:")
print(grid_search_nb.best_params_)




print(grid_search_lr.best_params_)

best_nb = MultinomialNB(**grid_search_nb.best_params_)
best_nb.fit(X_train2, y_train)

print(grid_search_nb.best_params_)

# Save the optimized classifiers
with open('classifier1.pkl', 'wb') as f:
    pickle.dump(best_lr, f)
    

# After fitting the GridSearchCV object
# results_lr = pd.DataFrame(grid_search_lr.cv_results_)

# # Get mean scores and the standard deviation of scores
# mean_train_scores = results_lr['mean_train_score']
# std_train_scores = results_lr['std_train_score']
# mean_test_scores = results_lr['mean_test_score']
# std_test_scores = results_lr['std_test_score']

# tested_C_values = results_lr['param_C'].astype(np.float64)
# penalties = results_lr['param_penalty']

# # Plotting for L1 penalty
# plt.figure(figsize=(10, 6))
# penalty = 'l1'
# penalty_mask = results_lr['param_penalty'] == penalty
# penalty_C_values = tested_C_values[penalty_mask]
# penalty_mean_train_scores = mean_train_scores[penalty_mask]
# penalty_mean_test_scores = mean_test_scores[penalty_mask]
# penalty_std_train_scores = std_train_scores[penalty_mask]
# penalty_std_test_scores = std_test_scores[penalty_mask]

# plt.errorbar(penalty_C_values, penalty_mean_train_scores, yerr=penalty_std_train_scores,
#              label=f'Training Score ({penalty.upper()})', fmt='o-', capsize=5)
# plt.errorbar(penalty_C_values, penalty_mean_test_scores, yerr=penalty_std_test_scores,
#              label=f'Validation Score ({penalty.upper()})', fmt='o-', capsize=5)

# plt.xscale('log')
# plt.xlabel('C (Inverse of Regularization Strength)')
# plt.ylabel('Mean ROC AUC Score')
# plt.title('Logistic Regression Hyperparameter Tuning Results: L1 Penalty')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting for L2 penalty
# plt.figure(figsize=(10, 6))
# penalty = 'l2'
# penalty_mask = results_lr['param_penalty'] == penalty
# penalty_C_values = tested_C_values[penalty_mask]
# penalty_mean_train_scores = mean_train_scores[penalty_mask]
# penalty_mean_test_scores = mean_test_scores[penalty_mask]
# penalty_std_train_scores = std_train_scores[penalty_mask]
# penalty_std_test_scores = std_test_scores[penalty_mask]

# plt.errorbar(penalty_C_values, penalty_mean_train_scores, yerr=penalty_std_train_scores,
#              label=f'Training Score ({penalty.upper()})', fmt='o-', capsize=5)
# plt.errorbar(penalty_C_values, penalty_mean_test_scores, yerr=penalty_std_test_scores,
#              label=f'Validation Score ({penalty.upper()})', fmt='o-', capsize=5)

# plt.xscale('log')
# plt.xlabel('C (Inverse of Regularization Strength)')
# plt.ylabel('Mean ROC AUC Score')
# plt.title('Logistic Regression Hyperparameter Tuning Results: L2 Penalty')
# plt.legend()
# plt.grid(True)
# plt.show()

results_nb = pd.DataFrame(grid_search_nb.cv_results_)

#print(results_nb)

# Get mean scores and the standard deviation of scores
mean_train_scores_nb = results_nb['mean_train_score']
std_train_scores_nb = results_nb['std_train_score']
mean_test_scores_nb = results_nb['mean_test_score']
std_test_scores_nb = results_nb['std_test_score']

#Extract the tested alpha values
tested_alpha_values = results_nb['param_alpha'].astype(np.float64)

# Plotting
plt.figure(figsize=(10, 6))

plt.errorbar(tested_alpha_values, mean_train_scores_nb, yerr=std_train_scores_nb,
             label='Training Score', fmt='o-', capsize=5)
plt.errorbar(tested_alpha_values, mean_test_scores_nb, yerr=std_test_scores_nb,
             label='Validation Score', fmt='o-', capsize=5)

plt.xscale('log')
plt.xlabel('Alpha (Smoothing Parameter)')
plt.ylabel('Mean ROC AUC Score')
plt.title('MultinomialNB Hyperparameter Tuning Results')
plt.legend()
plt.grid(True)


plt.show()



with open('classifier2.pkl', 'wb') as f:
    pickle.dump(best_nb, f)
