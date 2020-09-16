'''
This script uses Stratified K-folds to test a Random Forest model using training
data.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def random_forest_model_cv(x_data, y_data, number_estimators=100):
    '''
    A function that models data on Random Forest Classification using Stratified K-folds.

    Parameters
    ----------
    X : Feature training and validation set.
    y : Target training and validation set.
    number_estimators : Number of trees in forest, 100 if not specified

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

        random_forest = RandomForestClassifier(n_estimators=number_estimators)
        random_forest.fit(x_train, y_train)
        y_pred = random_forest.predict(x_val)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_auc_scores.append(roc_auc_score(y_val, random_forest.predict_proba(x_val)[:, 1]))

    print('Random Forest results:\n'
          f'Accuracy mean: {np.mean(accuracy_scores)},\n'
          f'Precision mean: {np.mean(precision_scores)},\n'
          f'Recall mean: {np.mean(recall_scores)},\n'
          f'F1 mean: {np.mean(f1_scores)},\n'
          f'ROC AUC: {np.mean(roc_auc_scores)}')
