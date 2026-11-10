# phase2_rag/day14_rag.py
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from google import genai
from google.genai import types

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"


def ask(prompt: str, system: str = "") -> str:
    contents = f"{system}\n\n{prompt}" if system else prompt
    r = _client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return r.text or "(no text returned)"


EMBED = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
db = chromadb.PersistentClient(path="chroma")


def get_col(name="coop_regs"):
    return db.get_or_create_collection(name, embedding_function=EMBED)


def chunk(text, size=800, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def ingest(pdf_path, col_name="coop_regs"):
    col = get_col(col_name)
    ids, docs, metas = [], [], []
    for p, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for j, c in enumerate(chunk(text)):
            if len(c.strip()) < 50:
                continue
            ids.append(f"p{p}-c{j}")
            docs.append(c)
            metas.append({"page": p, "source": pdf_path})
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def retrieve(q, k=6, col_name="coop_regs"):
    r = get_col(col_name).query(query_texts=[q], n_results=k)
    seen, kept = set(), []
    for d, m in zip(r["documents"][0], r["metadatas"][0]):
        key = d[:80]
        if key in seen:
            continue
        seen.add(key)
        kept.append((d, m))
    return kept


SYSTEM = (
    "You answer questions about cooperative regulations using ONLY the context. "
    "Cite pages like [p.3] after each fact. Do not repeat the [p.N] markers "
    "from the context in your prose. If the context does not contain the answer, "
    "reply exactly: I can't find that in the regulations."
)


def answer(q, k=6, col_name="coop_regs"):
    hits = retrieve(q, k, col_name)
    context = "\n\n".join(f"[p.{m['page']}] {d}" for d, m in hits)
    return ask(f"Context:\n{context}\n\nQuestion: {q}", system=SYSTEM), hits


if __name__ == "__main__":
    col = get_col()
    if col.count() == 0:
        print("chunks:", ingest("data/coop_act.pdf"))
    else:
        print("already ingested:", col.count())
    try:
        while (q := input("\nQuestion (blank to quit): ").strip()):
            text, hits = answer(q)
            print(text, "\n  sources:", [m["page"] for _, m in hits])
    except (KeyboardInterrupt, EOFError):
        print("\nbye")