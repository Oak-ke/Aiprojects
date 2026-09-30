# RAG Basics: Chunking Strategies for Agricultural Guidelines
This project evaluates document **chunking strategies** for Retrieval-Augmented Generation (RAG) pipelines. It benchmark naive fixed-size chunking against recursive semantic chunking using agricultural extension advisories from the Kenya Agricultural and Livestock Research Organization (KALRO).

## The RAG Challenge: Information Loss at Boundaries
When indexing long technical manuals into a vector database, embedding an entire multi-page document dilutes vector representations. However, slicing text naively into fixed character blocks introduces severe boundary cutoffs—often separating a crop disease name from its corresponding pesticide dosage or safety warning.

## Evaluated Chunking Strategies

1. **Fixed-Size Chunking (`CharacterTextSplitter`):**
   - Splits text strictly by character count regardless of natural language structure.
   - Fast, but frequently severs sentences, tables, and dosage instructions.

2. **Recursive Semantic Chunking (`RecursiveCharacterTextSplitter`):**
   - Recursively splits along structural separators (`\n\n`, `\n`, bullet points, spaces).
   - Keeps paragraphs, procedural steps, and safety warnings intact within single context blocks for accurate semantic retrieval.

## Tech Stack
* **Language:** Python
* **Text Processing:** LangChain Text Splitters (`langchain-text-splitters`)
* **Use Case:** Agricultural Extension Advisory Retrieval

## Project Structure
```text
├── chunking_demo.py       # Benchmark script comparing fixed vs. recursive chunking
├── requirements.txt       # Dependencies (langchain-text-splitters)
└── README.md              # Project documentation

```

### Setup & Usage
1. Install Dependencies
```Bash
pip install langchain-text-splitters
```
### 2. Run the Benchmark
```Bash
python chunking_demo.py
```
---

### What Does `chunk_overlap` Do?

`chunk_overlap` defines the number of characters (or tokens) that consecutive chunks **share in common**. It creates a sliding window across your document instead of hard, distinct cuts.

```text
Without Overlap (chunk_size=10, overlap=0):
[ Chunk 1: "Apply 15mL" ] [ Chunk 2: "per 20L water" ]

With Overlap (chunk_size=10, overlap=5):
[ Chunk 1: "Apply 15mL" ]
            [ Chunk 2: "15mL per 20L" ]
```