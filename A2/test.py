from sklearn.cluster import KMeans
import numpy as np
X = np.array([[0, 2], [5, 4], [-2, -4],[4, -3], [-4, 1], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)