# -------------------------------------------------------------------------
# AUTHOR: Brandon Diep
# FILENAME: pca.py
# SPECIFICATION: This program finds highest PC1 variance when removing a feature from the heart disease dataset
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Load the data
#--> add your Python code here

df = pd.read_csv("heart_disease_dataset.csv")


#Create a training matrix
#--> add your Python code here
df_features = df

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = df_features.shape[1]

# Run PCA for 9 features, removing one feature at each iteration
pc1_variances = []
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, 1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance = pca.explained_variance_ratio_[0]
    removed_feature = df_features.columns[i]
    pc1_variances.append((pc1_variance, removed_feature))


# Find the maximum PC1 variance
# --> add your Python code here
max_PC1variance, feature = max(pc1_variances)


#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {max_PC1variance} when removing {feature}")




