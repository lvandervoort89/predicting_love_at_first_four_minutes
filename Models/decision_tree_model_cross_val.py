'''
This script uses Stratified K-folds to test a Decision Tree model using training
data.
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def decision_tree_model_cv(x_data, y_data, max_d=None):
    '''
    A function that models data on Decision Tree Classifier using Stratified K-folds.

    Parameters
    ----------
    X : Feature training and validation set.
    y : Target training and validation set.
    max_d : Maximum depth of tree, None if not specified

    Returns
    -------
    Prints the average accuracy score, precision score, recall score, F1, and
    ROC AUC score of the folds.

    '''
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_ind, val_ind in skf.split(x_data, y_data):
        x_train, y_train = x_data.iloc[train_ind], y_data.iloc[train_ind]
        x_val, y_val = x_data.iloc[val_ind], y_data.iloc[val_ind]
        decision_tree = DecisionTreeClassifier(max_depth=max_d)
        decision_tree.fit(x_train, y_train)
        y_pred = decision_tree.predict(x_val)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_auc_scores.append(roc_auc_score(y_val, decision_tree.predict_proba(x_val)[:, 1]))

    print('Decision tree results:\n'
          f'Accuracy mean: {np.mean(accuracy_scores)},\n'
          f'Precision mean: {np.mean(precision_scores)},\n'
          f'Recall mean: {np.mean(recall_scores)},\n'
          f'F1 mean: {np.mean(f1_scores)},\n'
          f'ROC AUC: {np.mean(roc_auc_scores)}')
