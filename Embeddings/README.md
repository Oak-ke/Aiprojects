# Tractor Service Clustering 🚜

**Description:** 
This project helps us understand how AI embeddings work by translating text data into coordinates within a high-dimensional vector space and mathematically determining relationships using cosine similarity. Specifically, this code takes tractor service logs, generates 384-dimensional text embeddings, normalizes the associated repair prices, and fuses them into a single dataset. Finally, it uses the elbow method to find the optimal *K* and applies K-Means clustering to automatically group similar services together.

**Libraries and Packages Used:**
*   `sentence-transformers`: To generate 384-dimensional text embeddings using `all-MiniLM-L6-v2`.
*   `scikit-learn`: To normalize the prices and cluster the fused data using K-Means.
*   `numpy`: To fuse the text vectors and normalized prices together.

**How to Run:**
1. Install dependencies: `pip install sentence-transformers scikit-learn numpy`
2. Run the script: `python cluster_logs.py`