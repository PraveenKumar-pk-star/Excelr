import pandas as pd
import streamlit as st

train_df = pd.read_csv('/content/Titanic_train.csv')
test_df = pd.read_csv('/content/Titanic_test.csv')
train_df



test_df

train_df.head()

# b. Examine the features, their types, and summary statistics


train_df_info = train_df.info()
train_df_description = train_df.describe(include='all')
train_df_info

train_df_description

# c. Create visualizations

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
# Histogram for numerical variables
numerical_vars = ['Age', 'Fare', 'SibSp', 'Parch']
for i, var in enumerate(numerical_vars, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[var].dropna(), kde=True, bins=30)
    plt.title(f'Histogram of {var}')
plt.tight_layout()
plt.show()

# Box plots for numerical variables
plt.figure(figsize=(20, 15))

for i, var in enumerate(numerical_vars, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=train_df[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()

# Bar plots for categorical variables
categorical_vars = ['Survived', 'Pclass', 'Sex', 'Embarked']

plt.figure(figsize=(20, 15))

for i, var in enumerate(categorical_vars, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=train_df[var])
    plt.title(f'Count Plot of {var}')

plt.tight_layout()
plt.show()

# 2. Data Preprocessing

# a. Handle missing values (e.g., imputation)

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df = train_df.drop(columns=['Cabin'])

print(train_df.isnull().sum())

# b. Encode categorical variables

train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
print(train_df.head())

# 3. Model Building

# a. Build a logistic regression model using appropriate libraries (e.g., scikit-learn)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 4. Model Evaluation

# a. Evaluate the performance of the model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print(f'ROC-AUC score: {roc_auc:.2f}')
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 5. Interpretation

# a. Interpret the coefficients of the logistic regression model

import numpy as np
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
})
print(coef_df)