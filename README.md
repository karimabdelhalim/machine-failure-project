
# Machine Failure Classification Project

This project aims to classify machine failures using machine learning techniques.

## Overview

In this project, we build a classification model to predict machine failures based on various features collected from machines. We use a logistic regression model for classification and evaluate its performance using cross-validation ROC AUC scores.

## Data

The dataset used in this project consists of two CSV files:

- `train.csv`: Contains the training data with features and labels.
- `test.csv`: Contains the test data without labels.

The features include numerical and categorical variables related to machine operations.

## Model Building

We preprocess the data by encoding categorical variables using target encoding and perform feature engineering by transforming the 'Rotational speed [rpm]' feature using a logarithmic transformation.

We use logistic regression as our classification model due to its simplicity and interpretability. The model is trained on the training data and evaluated using cross-validation ROC AUC scores.

## Results

The trained model achieves an average ROC AUC score of [98%] across cross-validation folds.



## Author

[karim abdelhalim]
[youssef amine]
[mohamed stohy]
[amira gomaa ]


