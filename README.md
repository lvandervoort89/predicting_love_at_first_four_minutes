# Predicting Love at First Four Minutes

## **Objective:**
Build 2 binary classification models to predict 1) a match in a round of speed dating
and 2) a date after a speed dating event

## **Approach:**
Speed dating data is cleaned and aggregated into two data frames to build two separate classification models. The first model predicts whether a round of speed dating will end in a match. The features used in this model were related to how participants scored their dates based on characteristics like attractiveness, intelligence, and ambition. A feature was engineered that measured how in-sync the date was by finding the absolute value of the sum of the differences in how a participant rated their date and how the date rated the participant on each characteristic.  The second model predicts whether a match in a speed dating round results in a date after the speed dating event. The features used in this model were related to how each participant rated their interests in various categories, how much they go out (in general and on dates), and how they view themselves. Numerous features were engineered in order to categorize each activity.

## **Featured Techniques:**
- Feature Engineering & Selection
- Supervised Machine Learning
- Classification
- Logistic Regression
- Random Forest
- Decision Trees
- K-Nearest Neighbors
- SVM

## **Data:**
This [dataset](https://www.kaggle.com/annavictoria/speed-dating-experiment) was obtained from Kaggle.

## **Results Summary:**
Both classification models were optimized for ROC AUC score so that the end user could choose their tolerance to false positives (the model predicting a match/date and it not ending in one) since dating is a vulnerable and personal experience. The Logistic Regression model that predicted a match in a round of speed dating was optimized with a ROC AUC of 0.84.  The Random Forest model that predicted a date after a speed dating event was optimized with a ROC AUC of 0.79.
