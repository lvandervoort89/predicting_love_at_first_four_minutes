'''
This script uses Stratified K-folds to test a SVM model using training
data.
'''

from sklearn import svm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def svm_model_cv_with_ss(x_data, y_data, kernels='rbf'):
    '''
    A function that models data on SVM using Stratified K-folds WITH
    StandardScaler.

    Parameters
    ----------
    X : Feature training and validation set.
    y : Target training and validation set.
    kernels : Specifies the kernel type to be used in the algorithm.
    It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    or a callable. If none is given, ‘rbf’ will be used.

    Returns
    -------
    Prints the average accuracy score, precision score, recall score, and F1 score of the folds.

    '''
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_ind, val_ind in skf.split(x_data, y_data):
        x_train, y_train = x_data.iloc[train_ind], y_data.iloc[train_ind]
        x_val, y_val = x_data.iloc[val_ind], y_data.iloc[val_ind]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train.values)
        x_val_scaled = scaler.transform(x_val.values)

        svm_model = svm.SVC(kernel=kernels)
        svm_model.fit(x_train_scaled, y_train)
        y_pred = svm_model.predict(x_val_scaled)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))

    print('SVM results:\n'
          f'Accuracy mean: {np.mean(accuracy_scores)},\n'
          f'Precision mean: {np.mean(precision_scores)},\n'
          f'Recall mean: {np.mean(recall_scores)},\n'
          f'F1 mean: {np.mean(f1_scores)}')
