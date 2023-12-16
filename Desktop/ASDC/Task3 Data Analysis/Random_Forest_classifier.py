"""
Created on Thu Dec 14 08:10:11 2023

@author: AHMED YASSER
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load the dataset
df = pd.read_csv('C:/Users/xx6k/Desktop/ASDC/Task3 Data Analysis/ai4i2020.csv')
print(df.isnull().sum(axis=0))
print(df.dtypes)

# Data Preprocessing
# Convert categorical variables into numerical format
label_encoder = LabelEncoder()
df['Product ID'] = label_encoder.fit_transform(df['Product ID'])
df['Type'] = label_encoder.fit_transform(df['Type'])

# Handle missing values if any
# For simplicity, we'll use mean imputation for numerical features
#df.fillna(df.mean(), inplace=True)

# Data Splitting
selected_features = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[selected_features]
y = df['Machine failure']
# Normalize or scale numerical features (example: using StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Using a Random Forest classifier as an example
model = RandomForestClassifier(random_state=42)
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

# Visualize Feature Importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 4))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

