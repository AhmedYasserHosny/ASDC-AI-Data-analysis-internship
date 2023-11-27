# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:22:54 2023

@author: AHMED YASSER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog  # Importing filedialog from tkinter
# Create a Tkinter root window (it will not be shown)
root = Tk()
root.withdraw()  # Hide the main window

# Ask the user to select a file
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

# Check if the user selected a file
if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load the dataset
df = pd.read_csv(file_path)
print(df.head())
#printing some statistical information like first quartile and so on 
print(df.describe())
#print the dataset information (no.of rows and columns)
print(df.info())
#printing the total null value in each colum
#print(df.isnull().sum(axis=0))

##clean data :: numrical 
#--------------------------
dt=df.copy()
col_num=["City_Code_Patient","Bed Grade"]
dt[col_num]=dt[col_num].fillna(dt.mean())
#dt[col_num]=dt[col_num].fillna(dt.median())
print(dt.isnull().sum(axis=0)) #check there is a null value also after clean
print(dt.dtypes) #dtype property retun the type of each column


# 1. Admission Type Distribution
sns.countplot(x="Type of Admission" ,data=dt)
plt.figure(figsize=(8, 6))
sns.countplot(data=dt, x='Type of Admission')
plt.title('Distribution of Admission Types')
plt.xlabel('Type of Admission')
plt.ylabel('Count')
plt.show()

# 2. Demographics - Age Distribution
sns.countplot(x="Age" ,data=dt)
plt.figure(figsize=(12, 6))
sns.histplot(data=dt, x='Age', bins=30, kde=True)
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 3. Length of Stay vs. Severity of Illness
sns.countplot(x="Severity of Illness", data=dt)
plt.figure(figsize=(10, 6))
sns.boxplot(data=dt, x='Severity of Illness', y='Stay')
plt.title('Length of Stay vs. Severity of Illness')
plt.xlabel('Severity of Illness')
plt.ylabel('Length of Stay')
plt.show()

sns.countplot(x="Hospital_region_code" ,data=dt)

# 4. Length of Stay vs. Severity of Illness
sns.countplot(x="Severity of Illness" ,data=dt)
plt.figure(figsize=(10, 6))
sns.boxplot(data=dt, x='Severity of Illness', y='Stay')
plt.title('Length of Stay vs. Severity of Illness')
plt.xlabel('Severity of Illness')
plt.ylabel('Length of Stay')
plt.show()

# 5. Hospital Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=dt, x='Hospital_type_code')
plt.title('Distribution of Hospital Types')
plt.xlabel('Hospital Type Code')
plt.ylabel('Count')
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 7. Ward Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=dt, x='Ward_Type')
plt.title('Distribution of Ward Types')
plt.xlabel('Ward Type')
plt.ylabel('Count')
plt.show()

#encoding the alphptic records to can using IN Graph 
#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
col_obj=["Hospital_type_code","Hospital_region_code","Department","Ward_Type","Ward_Facility_Code","Type of Admission","Severity of Illness","Age"]
dt[col_obj]=dt[col_obj].astype('category')

for col in col_obj:
    dt[col]=dt[col].cat.codes
    print(dt.dtypes)
    
