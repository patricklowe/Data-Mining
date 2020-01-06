import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
# Load the data
df_1 = pd.read_csv( "specs/question_1.csv")
"""
Question 1
"""
#   Get the values from the dataframe
X = df_1.values
#   Apply the K-Means clustering on 3 clusters
kmeans = KMeans(n_clusters=3, n_init = 10, random_state=0).fit(X)
#   Gather the labels of cluster associated with each entry
k_labels = kmeans.labels_
#   Get the centroid of each cluster
k_centers = kmeans.cluster_centers_
#   Save the labels
df_1['cluster'] = k_labels
#   Plot the clusters and their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='red')
plt.title("Q1. K-Means (3 Clusters)")
#   Export the graph and the results to a pdf/csv
plt.savefig("output/question_1.pdf", bbox_inches='tight')
df_1.to_csv('output/question1_out.csv', index=False)

"""
Question 2
"""
#   Load the data
df_2 = pd.read_csv( "specs/question_2.csv")

# Discard name, type, manufacturer, rating
df_2 = df_2.drop('NAME', axis = 1)
df_2 = df_2.drop('TYPE', axis = 1)
df_2 = df_2.drop('MANUF', axis = 1)
df_2 = df_2.drop('RATING', axis = 1)

# K-Means
X = df_2.values
kmeans2 = KMeans(n_clusters = 5, n_init = 5, max_iter = 100, random_state = 0).fit(X)
k_labels2 = kmeans2.labels_
k_centers2 = kmeans2.cluster_centers_
df_2['config1'] = k_labels2
#   K-Means for 2nd configuration
kmeans3 = KMeans(n_clusters = 5, n_init = 100, max_iter = 100, random_state = 0).fit(X)
k_labels3 = kmeans3.labels_
k_centers3 = kmeans3.cluster_centers_
df_2['config2'] = k_labels3
# Create a df for comparison
comparison = df_2[['config1', 'config2']].copy()
print(comparison['config1'].value_counts())
print(comparison['config2'].value_counts())
# 3rd configuration of K-Means
kmeans4 = KMeans(n_clusters = 3, n_init = 100, max_iter = 100, random_state = 0).fit(X)
k_labels4 = kmeans4.labels_
k_centers4 = kmeans4.cluster_centers_
df_2['config3'] = k_labels4

comparison = df_2[['config1', 'config2', 'config3']].copy()
print(comparison)

print(comparison['config3'].value_counts())

# Export to a CSV
df_2.to_csv('output/question2_out.csv', index=False)

"""
Question 3
"""
#   Load the data
df_3 = pd.read_csv( "specs/question_3.csv")
df_3 = df_3.drop('ID', axis = 1)

# K-Means
X = df_3.values
kmeans5 = KMeans(n_clusters = 7, n_init = 5, max_iter = 100, random_state = 0).fit(X)
k_labels5 = kmeans5.labels_
k_centers5 = kmeans5.cluster_centers_
df_3['kmeans'] = k_labels5
#   Plot the clusters and their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=k_labels5)
plt.scatter(k_centers5[:, 0], k_centers5[:, 1], c='red')
plt.title("Q3. K-Means (7 Clusters)")
#   Export the graph and the results to a pdf/csv
plt.savefig("output/question_3_1.pdf", bbox_inches='tight')


#   Normalise X
x_Min = df_3['x'].min()
x_Max = df_3['x'].max()
df_3['x'] = (df_3['x'] - x_Min)/(x_Max-x_Min)
#   Normalize Y
y_Min = df_3['y'].min()
y_Max = df_3['y'].max()
df_3['y'] = (df_3['y'] - y_Min)/(y_Max-y_Min)

# Cluster using DBSCAN 0.04
X = df_3.iloc[:,0:2].values
clustering = DBSCAN(eps = 0.04, min_samples = 4).fit(X)
c_labels = clustering.labels_
df_3['dbscan1'] = c_labels
#   Plot the clusters and their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=c_labels)
plt.title("Q3. DBSCAN 1")
#   Export the graph and the results to a pdf/csv
plt.savefig("output/question_3_2.pdf", bbox_inches='tight')

# Cluster using DBSCAN 0.08
X = df_3.iloc[:,0:2].values
clustering = DBSCAN(eps = 0.08, min_samples = 4).fit(X)
c_labels = clustering.labels_
df_3['dbscan2'] = c_labels
#   Plot the clusters and their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=c_labels)
plt.title("Q3. DBSCAN 2")
#   Export the graph and the results to a pdf/csv
plt.savefig("output/question_3_3.pdf", bbox_inches='tight')
df_3.to_csv('output/question3_out.csv', index=False)

comparison = df_3[['dbscan1', 'dbscan2']].copy()
print(comparison['dbscan1'].value_counts())
print(comparison['dbscan2'].value_counts())




