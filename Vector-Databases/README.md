# Semantic Vector Search for Informal Retail (Duka) Inventory

This project demonstrates a localized **Vector Database** implementation using ChromaDB, designed to power semantic search for informal retail shops (dukas) across East Africa. It showcases how to leverage **HNSW (Hierarchical Navigable Small World)** indexing to deliver enterprise-grade AI search while accommodating the hardware constraints of the informal sector.

## The Business Challenge: Low-End Hardware & Messy Queries
Informal shopkeepers manage their businesses primarily on budget Android devices. Running heavy embedding models or executing complex text-matching algorithms locally on these devices leads to rapid battery drain and application crashes. 

Furthermore, inventory queries are rarely perfect English keywords. A customer or shopkeeper might search for *"flour for making ugali"* or *"unga ya sima"*, which a traditional SQL `LIKE` query would completely fail to match against a product listed as *"2kg Jogoo Maize Flour"*.

## The Solution: Edge-to-Cloud Vector Architecture
This architecture shifts the compute burden to the cloud:
1. **The Edge:** The lightweight mobile app captures the natural language query and sends it to the server.
2. **The Cloud (Vector DB):** ChromaDB converts the query into a high-dimensional vector.
3. **HNSW Retrieval:** Instead of scanning every product row by row $O(n)$, the database navigates the HNSW graph index to mathematically snap to the closest semantic product instantly.

Even if the query lacks exact keywords, the semantic meaning connects the intent to the correct inventory item.

## Tech Stack
* **Language:** Python
* **Vector Database:** ChromaDB (Local/In-Memory for demonstration)
* **Embedding Model:** `sentence-transformers` (`all-MiniLM-L6-v2` for lightweight, rapid vectorization)
* **Core Techniques:** HNSW Indexing, Semantic Search

## Project Structure
```text
├── duka_inventory.py       # Main ChromaDB vectorization and retrieval script
├── requirements.txt        # Project dependencies
└── README.md               # Architecture and usage documentation
```
## Setup & Usage

1. Install Dependencies
Ensure your virtual environment is active, then install ChromaDB and the required embedding models:

```Bash
pip install chromadb sentence-transformers
```
## 2. Run the Inventory Engine
```Bash
python duka_inventory.py
```