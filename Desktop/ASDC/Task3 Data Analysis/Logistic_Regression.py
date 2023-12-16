# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:28:30 2023

@author: AHMED YASSER
"""
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


# Load the dataset
df = pd.read_csv('C:/Users/xx6k/Desktop/ASDC/Task3 Data Analysis/ai4i2020.csv')
print(df.isnull().sum(axis=0))
print(df.dtypes)

# Data Preprocessing
# Convert categorical variables into numerical format
label_encoder_product_id = LabelEncoder()
df['Product ID'] = label_encoder_product_id.fit_transform(df['Product ID'])

label_encoder_type = LabelEncoder()
df['Type'] = label_encoder_type.fit_transform(df['Type'])

# Handle missing values if any
# For simplicity, we'll use mean imputation for numerical features
# df.fillna(df.mean(), inplace=True)

# Data Splitting
selected_features = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[selected_features]
y = df['Machine failure']
# Normalize or scale numerical features (example: using StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Using Logistic Regression as an example
model = LogisticRegression(random_state=42)
# Model Training
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)

# Print accuracy and other metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Extract the features of the first row and convert to the required format
first_row_features = pd.DataFrame({
    'Product ID': ['L47249'],
    'Type': ['L'],
    'Air temperature [K]': [298.9],
    'Process temperature [K]': [309],
    'Rotational speed [rpm]': [1410],
    'Torque [Nm]': [65],
    'Tool wear [min]': [8]
})

# Encode 'Product ID' and 'Type' using the corresponding label encoders
first_row_features['Product ID'] = label_encoder_product_id.transform(first_row_features['Product ID'])
first_row_features['Type'] = label_encoder_type.transform(first_row_features['Type'])

# Normalize or scale numerical features using StandardScaler
first_row_features_scaled = scaler.transform(first_row_features)
# Make prediction without scaling the features
y_pred_first_row = model.predict(first_row_features)
# Print the prediction
print(f'Machine Failure Prediction for the first row: {y_pred_first_row}')


