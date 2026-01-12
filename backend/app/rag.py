import os, time, json
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings

from .config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, VECTOR_DIR

client = OpenAI(api_key=OPENAI_API_KEY)

def _get_embedder():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

def load_or_create_store():
    if os.path.exists(VECTOR_DIR):
        return FAISS.load_local(VECTOR_DIR, _get_embedder(), allow_dangerous_deserialization=True)
    return None

def save_store(store: FAISS):
    store.save_local(VECTOR_DIR)

def add_chunks_to_store(chunks: list[str], source_name: str):
    docs = [Document(page_content=c, metadata={"source": source_name}) for c in chunks]
    store = load_or_create_store()
    if store is None:
        store = FAISS.from_documents(docs, _get_embedder())
    else:
        store.add_documents(docs)
    save_store(store)

def answer(query: str, k: int = 4):
    store = load_or_create_store()
    if store is None:
        return {"answer": "No documents indexed yet. Please ingest PDFs first.", "sources": []}

    t0 = time.time()
    results = store.similarity_search(query, k=k)
    ctx = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(results)])

    prompt = f"""You are a helpful study assistant.
Use the context to answer. If not in context, say you don't know.
Return a short answer + bullet points, and cite sources like [1], [2].

Context:
{ctx}

Question: {query}
"""

    resp = client.responses.create(model=CHAT_MODEL, input=prompt)
    latency = time.time() - t0

    sources = [
        {"idx": i+1, "source": d.metadata.get("source", ""), "preview": d.page_content[:180]}
        for i, d in enumerate(results)
    ]

    os.makedirs("logs", exist_ok=True)
    with open("logs/qa_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "query": query,
            "k": k,
            "latency_s": round(latency, 3),
            "sources": sources
        }, ensure_ascii=False) + "\n")

    return {"answer": resp.output_text, "sources": sources, "latency_s": latency}
