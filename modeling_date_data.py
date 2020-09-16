'''
This scripts loads the date_data and uses a Logistic Regression model to predict a
match in a round of speed dating.
'''
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def separate_features_and_target(dataframe):
    '''
    Returns 2 dataframes where features contains only the features for the
    model and target contains the targets.
    '''

    features_match, target_match = dataframe.drop('match', axis=1), dataframe['match']

    return features_match, target_match

def train_test_split_data(date_data):
    '''
    Return train and test dataframes.
    '''

    feature_match, target_match = separate_features_and_target(date_data)

    x_train, x_test, y_train, y_test = train_test_split(feature_match, target_match, test_size=.2)

    return x_train, x_test, y_train, y_test

def final_logistic_regression_model(x_train, x_test, y_train, y_test):
    '''
    Takes in a dataframe and calls other functions to split the data into train and test sets.
    Models data using a logistic regression model to predict match after a round of speed dating.
    Prints accuracy, recall, precision, f1, and ROC AUC. Pickles the model for future use.
    '''
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.values)
    x_test_scaled = scaler.transform(x_test.values)

    log_reg = LogisticRegression()
    log_reg.fit(x_train_scaled, y_train)
    y_pred = log_reg.predict(x_test_scaled)

    model_accuracy_score = accuracy_score(y_test, y_pred)
    model_precision_score = precision_score(y_test, y_pred)
    model_recall_score = recall_score(y_test, y_pred)
    model_f1_score = f1_score(y_test, y_pred)
    model_roc_auc_score = roc_auc_score(y_test,
                                        log_reg.predict_proba(x_test_scaled)[:, 1])

    # Print model results
    print('Logistic Regression Results:\n'
          f'Accuracy: {model_accuracy_score},\n'
          f'Precision: {model_precision_score},\n'
          f'Recall: {model_recall_score},\n'
          f'F1: {model_f1_score},\n'
          f'ROC AUC: {model_roc_auc_score}')

    # Pickle model
    with open('log_reg.pkl', 'wb') as f:
        pickle.dump(log_reg, f)

def main():
    '''
    Loads in the date_data, separates the features and target, separates the data
    into train-test-split, and uses a Logistic Regression model. Prints the results and
    pickles the model.
    '''

    # Load in data
    date_data = pd.read_csv('date_data.csv')

    # Call internal functions to this script
    x_train, x_test, y_train, y_test = train_test_split_data(date_data)
    final_logistic_regression_model(x_train, x_test, y_train, y_test)

main()
