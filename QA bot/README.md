# Coop Regs QA Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
Kenya's cooperative regulations using **only** the contents of a local PDF.
Answers are cited by page (`[p.3]`), and the bot refuses to answer when the
retrieved context doesn't cover the question.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Vector store:** ChromaDB (persistent, `chroma/`)
- **LLM:** Google Gemini `gemini-2.5-flash` (free tier)
- **Data source:** `data/coop_act.pdf` — nothing else

---

## How it works

```
data/coop_act.pdf
   │  pypdf → page text → 800-char chunks (150-char overlap)
   ▼
all-MiniLM-L6-v2 embeddings (local)
   ▼
chroma/  ◄──── retrieve(q, k=6)
                    │
        SYSTEM prompt + [p.N] chunks + question
                    ▼
          Gemini API (gemini-2.5-flash)
                    ▼
            answer with [p.N] citations
```

The PDF is read **once** at ingest. After that, each question does one vector
search against `chroma/` plus one Gemini call. Gemini never sees the PDF or
the vector store — only the retrieved chunks in the prompt.

---

## Requirements

- Python 3.10+
- A free Gemini API key: <https://aistudio.google.com/app/apikey>

```powershell
python -m pip install -U pypdf chromadb sentence-transformers google-genai
```

> If you also use `langchain-google-genai` in the same venv (it pins
> `google-genai<3.0`), install `python -m pip install "google-genai>=2.20,<3.0"`
> to avoid the conflict.

First run downloads the `all-MiniLM-L6-v2` model (~90 MB) into
`~/.cache/huggingface/` — one-time, needs internet.

---

## Setup

### 1. Put the PDF in place

```
QA bot/
  data/
    coop_act.pdf       ← the only document the bot knows about
```

### 2. Provide the API key

**Current shell only (simplest):**
```powershell
$env:GEMINI_API_KEY = "AIza...your-key..."
```

**Persist for future sessions:**
```powershell
setx GEMINI_API_KEY "AIza...your-key..."
```
Then **open a new terminal** — the current one won't see it.

**Project-isolated `.env` (recommended):**
```powershell
python -m pip install python-dotenv
```
Create `.env` in the project root:
```
GEMINI_API_KEY=AIza...your-key...
```
Add at the top of `QA_bot.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```
Add `.env` to `.gitignore` — **never commit API keys.**

Verify the key reaches Python:
```powershell
python -c "import os; print('GEMINI_API_KEY' in os.environ)"
```
Should print `True`.

---

## Usage

```powershell
python QA_bot.py
```

First run ingests the PDF (slow — embeds every chunk). Later runs detect
`chroma/` already has data and skip re-ingestion.

```
chunks: 214
Question (blank to quit): What are the duties of a cooperative treasurer?
According to the regulations, the treasurer must ... [p.14] ... [p.15]
  sources: [14, 15, 22, 31]
```

- Blank line → quit
- Answer ending in `I can't find that in the regulations.` → the retrieved
  chunks didn't contain the answer (by design; the bot won't hallucinate)

---

## Project layout

```
QA bot/
  QA_bot.py          ← the bot
  data/
    coop_act.pdf     ← the only knowledge source
  chroma/            ← vector store (created on first ingest; safe to delete)
  .env               ← optional, gitignored
```

If you swap the PDF, either bump the collection name in `get_col()` (e.g.
`coop_regs_v2`) or delete `chroma/` and re-run once — otherwise old and new
chunks mix in the index.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'GEMINI_API_KEY'` | Env var not set in this shell | `$env:GEMINI_API_KEY = "..."` in the **same** window, or `setx` + new terminal |
| `ClientError: 400 API key not valid` | Wrong/revoked key | Regenerate at AI Studio |
| `ClientError: 429 RESOURCE_EXHAUSTED` | Free-tier quota hit | Wait ~1 min, or set `MODEL = "gemini-2.0-flash"` |
| `FileNotFoundError: ...coop_act.pdf` | PDF missing or wrong cwd | Run from `QA bot/`; PDF at `data/coop_act.pdf` |
| `ModuleNotFoundError: pypdf` / `chromadb` / `google` | Deps missing in this venv | `python -m pip install -U pypdf chromadb sentence-transformers google-genai` |
| First run takes minutes | Embedding every chunk locally | Expected. Later runs skip it if `chroma/` exists |
| `pip` warning: `langchain-google-genai requires google-genai<3.0` | Shared venv pin conflict | `python -m pip install "google-genai>=2.20,<3.0"` |

---

## Notes & limitations

- **Stateless across questions** — each question retrieves fresh chunks; no
  conversation memory.
- **Page numbers are PDF page numbers**, not printed page numbers. If the PDF
  has cover pages, citations will be offset.
- **English-biased embedding model.** For a Swahili/English-mixed act, swap
  `model_name` to `paraphrase-multilingual-MiniLM-L12-v2`.
- **Free-tier Gemini inputs may be used to improve Google products.** Don't
  feed it confidential material if that matters.
- **Temperature is 0** for stable citations. Raising it makes answers more
  varied but less reliably cited.

---

## Customizing

- **Different PDF:** change the path in `ingest(...)` and bump the collection
  name in `get_col()` so old and new indexes don't mix.
- **Retrieval:** `k` in `retrieve()` / `answer()` (default 6); `size` and
  `overlap` in `chunk()` (default 800 / 150).
- **Model:** change `MODEL` at the top of `QA_bot.py`. Free options:
  `gemini-2.5-flash`, `gemini-2.0-flash`.

---


---