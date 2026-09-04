# Offline-First Cooperative Registry Analyzer

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline designed for resource-constrained or offline environments. This project processes agricultural cooperative registries and answers field queries using a local vector database and a quantized LLM (`TinyLlama`), requiring zero cloud APIs or active internet connectivity.

---

## Prerequisites & Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
Install the required local dependencies:

Bash
pip install pandas langchain-text-splitters langchain-huggingface langchain-chroma torch accelerate transformers sentence-transformers
How the Code Works
The script (app.py) executes five core stages entirely on your local machine:

1. Dataset Generation (Mock Kaggle Data)

The script constructs a Pandas DataFrame containing structured agricultural data (districts, seasons, crops, soil types, irrigation methods, and yields).

It iterates through the rows, converting structured tabular rows into natural language field reports (e.g., "In 2024, the Mangalore cooperative recorded a Kharif harvest...") and merges them into a continuous text document.

2. Text Chunking

Because large texts exceed a small LLM's memory window, RecursiveCharacterTextSplitter breaks the continuous document into bite-sized chunks (256 characters with a 30-character overlap) to prevent cutting sentences awkwardly.

3. Local Embeddings & Vector Storage

The chunks are passed through the lightweight all-MiniLM-L6-v2 embedding model (langchain_huggingface), which converts text into mathematical vectors representing semantic meaning.

These vectors are stored locally in ChromaDB (langchain_chroma) inside a local folder (./coop_registry_db), acting as an offline search index.

4. Local LLM Initialization

The script downloads and loads TinyLlama/TinyLlama-1.1B-Chat-v1.0 via Hugging Face Transformers (transformers).

The model runs locally on standard computer memory/hardware without needing an external API key.

5. Query and Retrieval

When a query is executed (e.g., "What crops are grown using drip irrigation in Mangalore?"), the vector database performs a similarity search to retrieve only the top 2 most relevant text chunks.

The script injects those chunks as context into a prompt and hands it to TinyLlama, which generates a precise, context-bound answer directly in your terminal.

Running the Project
Execute the script from your terminal inside the virtual environment:

Bash
python app.py