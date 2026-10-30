import chromadb
from chromadb.utils import embedding_functions

def build_duka_catalog():
    # 1. Initialize local ChromaDB client (in-memory for this demo)
    client = chromadb.Client()

    # 2. Define an embedding model
    # SentenceTransformers are lightweight and excellent for semantic mapping
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # 3. Create a collection (an index/table in Vector DB terms)
    # Chroma defaults to HNSW indexing under the hood for blazing fast retrieval
    collection = client.create_collection(
        name="duka_inventory",
        embedding_function=sentence_transformer_ef
    )

    # Mock Duka Inventory
    items = [
        "2kg Jogoo Maize Flour (Unga)",
        "500g Ndovu Baking Powder",
        "1L Fresh Fri Cooking Oil (Mafuta)",
        "1kg Kabras Sugar (Sukari)",
        "250g Ketepa Tea Leaves (Majani)"
    ]
    
    # Vector DBs require unique IDs for every chunk/item
    ids = ["item_1", "item_2", "item_3", "item_4", "item_5"]

    print("Embedding and storing inventory into ChromaDB using HNSW indexing...")
    collection.add(
        documents=items,
        ids=ids
    )
    print("Catalog stored successfully.\n")

    # Semantic Search: Querying with intent, not just exact keywords
    query = "flour for making ugali"
    print(f"Searching for: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=2 # Retrieve the top 2 closest semantic matches
    )

    print("\n--- Search Results ---")
    for i, doc in enumerate(results["documents"][0]):
        # Distance represents how far apart the vectors are. Lower is better.
        distance = results['distances'][0][i]
        print(f"Match {i+1}: {doc} (Distance: {distance:.4f})")

if __name__ == "__main__":
    build_duka_catalog()