import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ingest import load_pdf_text, chunk_text
from .rag import add_chunks_to_store, answer

app = FastAPI(title="LLM-Powered Study Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        with open(path, "wb") as f:
            f.write(content)

        text = load_pdf_text(path)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF (it may be scanned). Try a text-based PDF."
            )

        chunks = chunk_text(text)
        add_chunks_to_store(chunks, source_name=file.filename)

        return {"file": file.filename, "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        # 开发期：把错误信息返回，方便调试
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")

@app.post("/chat")
async def chat(payload: dict):
    query = payload.get("query", "").strip()
    k = int(payload.get("k", 4))
    return answer(query, k=k)
