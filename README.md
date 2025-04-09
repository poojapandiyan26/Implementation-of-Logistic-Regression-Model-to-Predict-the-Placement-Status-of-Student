# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Cleaning: Dropped irrelevant columns like sl_no and salary to avoid noise and data leakage.

2.Encoding Categorical Features: Used Label Encoding to convert categorical data like gender and education into numerical values.

3.Feature Scaling: Applied StandardScaler to normalize the feature values for better model performance.

4.Training: Trained a Logistic Regression model on the scaled training set using sklearn.

5.Evaluation: Assessed model accuracy and performance using confusion matrix and classification report.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:Pooja sri P
RegisterNumber:212224230197
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("//content//Placement_Data.csv")
data.drop(['sl_no', 'salary'], axis=1, inplace=True)

label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
## Output:
![image](https://github.com/user-attachments/assets/2267f920-b6d0-4ad7-ac82-e9d0362f1ed2)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
