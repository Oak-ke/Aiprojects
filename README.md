# Applied AI Projects for East Africa

Small, focused prototypes that apply LLMs and machine learning to real problems in East African health, agriculture and informal finance: code-switched Swahili and Sheng text, low-end hardware, patchy connectivity and sensitive data.

Each folder is self-contained, with its own README and `requirements.txt`.

## Health

| Project | What it does | Key techniques |
| --- | --- | --- |
| [CHW Triage Assistant](chw-triage-ai) | Keeps an LLM within protocol for community health workers and forces escalation on danger signs, with a 22-case safety evaluation | Persona + hard-bound prompting, red-flag recall, adversarial tests |
| [Code-switched clinic notes pipeline](Prompt-chaining) | Normalises Swahili–Sheng–English intake logs, then outputs validated JSON | Prompt chaining, Pydantic schema enforcement, Gemini |
| [Clinical document parser](Document_parser) | Extracts structured data from scanned clinical forms while keeping layout | Multimodal (vision) parsing, Gemini |
| [Clinical log extractor](Prompt%20Engineering) | Pulls structured fields from messy clinic logs, model-agnostic for offline use | Few-shot chain-of-thought, regex anchoring |

## Agriculture & cooperatives

| Project | What it does | Key techniques |
| --- | --- | --- |
| [Offline RAG for cooperative registries](Offline-RAG-Coop-Registry) | Answers field queries over cooperative records with no internet or cloud API | Local RAG, ChromaDB, all-MiniLM-L6-v2, TinyLlama |
| [Chunking benchmark](Chunking) | Compares fixed vs recursive chunking on KALRO extension advisories | RAG chunking strategy, overlap |
| [Tractor job pricing](C-O-T) | Reliable multi-step cost calculation for mechanisation services | Chain-of-thought prompting |
| [Tractor service clustering](Embeddings) | Groups service logs by meaning and price | Sentence embeddings, K-means, elbow method |

## Informal economy & language equity

| Project | What it does | Key techniques |
| --- | --- | --- |
| [Semantic Dilution](Semantic-Dilution) | Measures the "tokenization tax": Swahili costs 2.22 tokens/word vs 1.00 for English on GPT-4's tokenizer | tiktoken, tokens-per-word analysis |
| [Micro-loan extractor](Pydantic) | Turns informal SMS/WhatsApp loan messages into typed records | Structured outputs, Pydantic, LangChain |
| [Duka inventory search](Vector-Databases) | Semantic product search for informal shops ("unga ya sima" → maize flour) | ChromaDB, HNSW indexing |
| [M-PESA message classifier](Attention) | Classifies transaction messages and visualises attention weights | Self-attention from scratch, PyTorch |
| [Determinism tests](Determinism) | Shows how temperature and top-p change outputs for triage vs marketing text | Sampling parameters |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
cd <project-folder>
pip install -r requirements.txt
```

Projects that call Gemini need an API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"          # macOS/Linux
$env:GOOGLE_API_KEY="your_api_key_here"            # Windows PowerShell
```

## Author

Evan Mwami Selunani · MSc Artificial Intelligence (in progress) · [nobaloak.co.ke](https://nobaloak.co.ke) · [LinkedIn](https://www.linkedin.com/in/evan-se-lunani)
