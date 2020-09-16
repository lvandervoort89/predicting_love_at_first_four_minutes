'''
This script uses Stratified K-folds to test a Logisitic Regression model using 
training data.
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def log_reg_model_cv_with_ss(x_data, y_data, c_value=1000):
    '''
    A function that models data on Logisitc Regression using Stratified K-folds WITH
    StandardScaler.

    Parameters
    ----------
    X : Feature training and validation set.
    y : Target training and validation set.
    c_value : C parameter, default is 1000

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

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train.values)
        x_val_scaled = scaler.transform(x_val.values)

        log_reg = LogisticRegression(C=c_value)
        log_reg.fit(x_train_scaled, y_train)
        y_pred = log_reg.predict(x_val_scaled)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_auc_scores.append(roc_auc_score(y_val, log_reg.predict_proba(x_val_scaled)[:, 1]))

    print('Logistic results:\n'
          f'Accuracy mean: {np.mean(accuracy_scores)},\n'
          f'Precision mean: {np.mean(precision_scores)},\n'
          f'Recall mean: {np.mean(recall_scores)},\n'
          f'F1 mean: {np.mean(f1_scores)},\n'
          f'ROC AUC: {np.mean(roc_auc_scores)}')
