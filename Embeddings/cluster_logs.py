from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Our sample tractor service logs
service_logs = [
    "Changed engine oil",
    "Replaced engine oil and filter",
    "Fixed hydraulic valve leak"
]

# 3. Generate the embeddings
embeddings = model.encode(service_logs)


# Sample prices for our 3 logs (needs to be a 2D array for the scaler)
prices = np.array([[200], [250], [1500]])

# 1. Normalize the prices between 0 and 1
scaler = MinMaxScaler()
normalized_prices = scaler.fit_transform(prices)

# 2. Combine text embeddings with the normalized price
# np.hstack places the arrays side-by-side
fused_data = np.hstack((embeddings, normalized_prices))



# Based on the Elbow Method, let's assume we settled on 2 clusters for our small dataset
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(fused_data)

# The algorithm assigns each log a cluster number (e.g., 0 or 1)
cluster_assignments = kmeans.labels_

for log, label in zip(service_logs, cluster_assignments):
    print(f"Cluster {label}: {log}")