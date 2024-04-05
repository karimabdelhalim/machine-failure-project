import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from category_encoders import TargetEncoder
#import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


train=pd.read_csv('/app/train.csv')
test=pd.read_csv('/app/test.csv')

train_set=train.drop('Machine failure',axis=1 )

y=train['Machine failure']

train_set.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


correlation_matrix = train_set.corr(numeric_only=True)


categorical_columns = train_set.select_dtypes(include=['object']).columns
mode_values = train_set[categorical_columns].mode().iloc[0]
encoder = TargetEncoder(cols=categorical_columns, handle_unknown='value')
train_set_encoded = encoder.fit_transform(train_set, y)
test_set_encoded = encoder.transform(test)


for column in train_set_encoded.columns:
    value_counts = train_set_encoded[column].value_counts()
    print(f"Value counts for column '{column}':")
    print(value_counts)
    print()

train_set_encoded['Rotational speed [rpm]'] = np.log1p(train_set['Rotational speed [rpm]'])

test_set_encoded['Rotational speed [rpm]'] = np.log1p(train_set['Rotational speed [rpm]'])

model = LogisticRegression(max_iter=1000)
cv_scores_roc_auc = cross_val_score(model, train_set_encoded, y, cv=kfold, scoring='roc_auc')


print("Cross-validation ROC AUC scores:", cv_scores_roc_auc)
print("Mean ROC AUC score:", np.mean(cv_scores_roc_auc))


model.fit(train_set_encoded, y)
probabilities = model.predict_proba(test_set_encoded)
print(probabilities)