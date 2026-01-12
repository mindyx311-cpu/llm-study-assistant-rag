# LLM Study Assistant (RAG)

RAG-based LLM study assistant that answers course questions over PDFs/notes using embeddings + vector search.
Built with **FastAPI** backend and **React** chat UI.

## Features
- PDF ingestion → text extraction → chunking
- Embeddings + vector search (FAISS)
- RAG answering with source citations
- FastAPI REST API (`/ingest`, `/chat`)
- Lightweight logging (JSONL) for prompt/model comparisons
- (Planned) Simple evaluation script for retrieval hit-rate

## Tech Stack
- Backend: Python, FastAPI, LangChain, OpenAI, FAISS
- Frontend: React (Vite)
- Vector Store: FAISS (local)

## Project Structure
```text
llm-study-assistant-rag/
backend/ # FastAPI server + RAG pipeline
frontend/ # React chat UI
scripts/ # eval + utilities
sample_data/ # sample doc instructions

## Quickstart (Backend)
> Requires Python 3.11+

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# set key
cp .env.example .env
uvicorn app.main:app --reload --port 8000
