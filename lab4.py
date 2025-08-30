import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances

# K-Means Clustering (time measurement with 10,000 points)
data = np.random.rand(10000, 2) * 100

kmeans = KMeans(n_clusters=5, random_state=42)

start_time = time.time()
kmeans.fit(data)
end_time = time.time()

print(f"Time taken by K-means to find clusters: {end_time - start_time:.4f} seconds")
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Mini-Batch K-Means (different batch sizes)
batch_sizes = [100, 300, 500, 1000, 1500]
times = []

for batch_size in batch_sizes:
    minibatch_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=batch_size, random_state=42)

    start_time = time.time()
    minibatch_kmeans.fit(data)
    end_time = time.time()

    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"Time taken with batch size {batch_size}: {time_taken:.4f} seconds")

best_batch_size = batch_sizes[np.argmin(times)]
print(f"\nBest batch size: {best_batch_size}")

# K-Means with Visualization (3 clusters, 1,000 points)
data = np.random.rand(1000, 2) * 100
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title("K-Means Clustering")
plt.show()

# K-Means++ with Visualization (4 clusters, 1,000 points in range 0-200)
data = np.random.rand(1000, 2) * 200
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_plus.fit(data)

plt.scatter(data[:, 0], data[:, 1], c=kmeans_plus.labels_, cmap='viridis')
plt.scatter(kmeans_plus.cluster_centers_[:, 0], kmeans_plus.cluster_centers_[:, 1], s=300, c='red')
plt.title("K-Means++ Clustering")
plt.show()

# Agglomerative Clustering on Iris Dataset
iris = load_iris()
data = iris.data

agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.title("Agglomerative Clustering on Iris Dataset")
plt.show()

# Custom K-Medoids Implementation
class KMedoids:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        medoids = X[medoid_indices].copy()

        distances = pairwise_distances(X)

        for _ in range(100):  # max iterations
            # Assign each point to closest medoid
            medoid_distances = pairwise_distances(X, medoids)
            labels = np.argmin(medoid_distances, axis=1)

            # Update medoids
            new_medoids = []
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                if len(cluster_points) == 0:
                    new_medoids.append(medoids[k])
                    continue

                min_cost = float('inf')
                best_medoid_idx = cluster_points[0]

                for candidate_idx in cluster_points:
                    cost = np.sum(distances[candidate_idx, cluster_points])
                    if cost < min_cost:
                        min_cost = cost
                        best_medoid_idx = candidate_idx

                new_medoids.append(X[best_medoid_idx])

            new_medoids = np.array(new_medoids)

            if np.allclose(medoids, new_medoids):
                break

            medoids = new_medoids

        # Final assignment
        medoid_distances = pairwise_distances(X, medoids)
        self.labels_ = np.argmin(medoid_distances, axis=1)

        return self

# Run K-Medoids on Iris dataset
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(data)

plt.scatter(data[:, 0], data[:, 1], c=kmedoids.labels_, cmap='viridis')
plt.title("K-Medoids Clustering on Iris Dataset")
plt.show()
