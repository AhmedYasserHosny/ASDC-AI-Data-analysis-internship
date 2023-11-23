# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:50:44 2023

@author: AHMED YASSER
"""
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# Extract relevant features for segmentation
X = df[['Age', 'Annual Income (k$)']]
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("----------------------------------------------------------------------")
print(df.head())

# Standardize the data
#scaler1 = MinMaxScaler()
#x_Scaled= scaler1.fit_transform(X)

# Determine the number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
# Plot the Elbow method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()
# Based on the Elbow method, choose the optimal number of clusters
k = 3
# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled) #train & examination data
# Get centroid values
centroids = kmeans.cluster_centers_
# Print and/or use centroid values as needed
print("Centroid Values:")
#print(centroids)

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],color='purple',marker='*',label='centroid')
plt.title('Customer Segmentation with KMeans')
plt.xlabel('Standardized Age')
plt.ylabel('Standardized Annual Income')
plt.legend()
plt.show()
