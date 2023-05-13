#-------------------------------------------------------------------------
# AUTHOR: Pablo Duenas
# FILENAME: clustering.py
# SPECIFICATION: clustering analysis on a dataset using KMeans algorithm
# FOR: CS 4210- Assignment #5
# TIME SPENT: < 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.iloc[:, :-1]

#run kmeans testing different k values from 2 until 20 clusters
silhouette_scores = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    score = silhouette_score(X_training, kmeans.labels_)
    silhouette_scores.append(score)

#find which k maximizes the silhouette_coefficient
best_k = np.argmax(silhouette_scores) + 2

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(range(2, 21), silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient vs number of clusters')
plt.show()

#reading the test data (clusters) by using Pandas library
test_data = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels
labels = np.array(df.iloc[:, -1]).reshape(1, -1)[0]

#fit KMeans with the best k value on the training data
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(X_training)

#To see this message, remember to close the plt graph
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__()) 
