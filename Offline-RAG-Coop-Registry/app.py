import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

# 1. Mock dataset generation (mimics loading a Kaggle dataset)
data = {
    'Year': [2024, 2024, 2025, 2025],
    'District': ['Mangalore', 'Udupi', 'Mangalore', 'Nairobi'],
    'Season': ['Kharif', 'Rabi', 'Kharif', 'Long Rains'],
    'Crop': ['Maize', 'Paddy', 'Millets', 'Grevillea robusta'],
    'Soil_Type': ['Laterite', 'Alluvial', 'Red Soil', 'Loam'],
    'Irrigation_Type': ['Drip', 'Rainfed', 'Drip', 'Manual'],
    'Yield': [45, 60, 30, 50]
}
df = pd.DataFrame(data)

documents = []
for index, row in df.iterrows():
    log = f"In {row['Year']}, the {row['District']} cooperative recorded a {row['Season']} harvest. " \
          f"The primary crop was {row['Crop']} grown in {row['Soil_Type']} soil using {row['Irrigation_Type']} irrigation. " \
          f"The total yield was {row['Yield']} units."
    documents.append(log)
    
full_registry_text = " ".join(documents)

# 2. Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=30
)
chunks = splitter.create_documents([full_registry_text])
print(f"Split registry into {len(chunks)} chunks.")

# 3. Create local vector embeddings & database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./coop_registry_db"
)

# 4. Initialize local LLM pipeline
llm_pipeline = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    device_map="auto", 
    max_new_tokens=150
)

# 5. Query the database and generate response
query = "What crops are grown using drip irrigation in Mangalore?"
retrieved_docs = vector_db.similarity_search(query, k=2)
context = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer concisely based on the context:"
response = llm_pipeline(prompt)
print("\n--- AI Response ---")
print(response[0]['generated_text'])