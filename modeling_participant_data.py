'''
This scripts loads the date_data and uses a Random Forest model to predict a
date after a speed dating event.
'''
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def separate_features_and_target_participants(participant_data):
    '''
    Returns 2 dataframes where features contains only the features for the
    model and target contains the targets for the participant data.
    '''

    features_match = participant_data.drop('date_after', axis=1)
    target_match = participant_data['date_after']

    return features_match, target_match

def train_test_split_data(participant_data):
    '''
    Return train and test dataframes.
    '''

    features_match, target_match = separate_features_and_target_participants(participant_data)

    x_train, x_test, y_train, y_test = train_test_split(features_match, target_match, test_size=.2)

    return x_train, x_test, y_train, y_test

def final_random_forest_model(x_train, x_test, y_train, y_test):
    '''
    Takes in a dataframe and calls other functions to split the data into train and test sets.
    Models data using a random forest model to predict a date after a speed dating event.
    Prints accuracy, recall, precision, f1, and ROC AUC. Pickles the model for future use.
    '''
    random_forest = RandomForestClassifier(n_estimators=90, max_features='sqrt',
                                           class_weight='balanced')
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)

    model_accuracy_score = accuracy_score(y_test, y_pred)
    model_precision_score = precision_score(y_test, y_pred)
    model_recall_score = recall_score(y_test, y_pred)
    model_f1_score = f1_score(y_test, y_pred)
    model_roc_auc_score = roc_auc_score(y_test,
                                        random_forest.predict_proba(x_test)[:, 1])

    # Print model results
    print('Random Forest Results:\n'
          f'Accuracy: {model_accuracy_score},\n'
          f'Precision: {model_precision_score},\n'
          f'Recall: {model_recall_score},\n'
          f'F1: {model_f1_score},\n'
          f'ROC AUC: {model_roc_auc_score}')

    # Pickle model
    with open('random_forest.pkl', 'wb') as f:
        pickle.dump(random_forest, f)

def main():
    '''
    Loads in the participant_data, separates the features and target, separates the data
    into train-test-split, and uses a Random Forest model. Prints the results and
    pickles the model.
    '''

    # Load in data
    participant_data = pd.read_csv('participant_data.csv')

    # Call internal functions to this script
    x_train, x_test, y_train, y_test = train_test_split_data(participant_data)
    final_random_forest_model(x_train, x_test, y_train, y_test)

main()
