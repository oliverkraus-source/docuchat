"""HTTP API + web UI.

Run:  uvicorn docuchat.api:app --reload    then open http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import LLMNotConfigured, answer_question
from .ingest import chunk_file
from .search import BM25Index

DATA_DIR = Path(os.environ.get("DOCUCHAT_DATA", "data"))
INDEX_PATH = DATA_DIR / "index.json"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC = Path(__file__).parent / "static"
ALLOWED = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="DocuChat", version="0.1.0")
state = {"index": BM25Index.load(INDEX_PATH)}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(state["index"].chunks)}


@app.get("/api/documents")
def list_documents():
    counts: dict[str, int] = {}
    for chunk in state["index"].chunks:
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile):
    name = Path(file.filename or "").name          # strip any directory parts
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"Only {', '.join(sorted(ALLOWED))} files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / name
    target.write_bytes(data)
    try:
        new_chunks = chunk_file(target)
    except Exception as exc:  # unreadable / corrupt file
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read {name}: {exc}") from exc

    # Re-uploading a file replaces its old chunks instead of duplicating them.
    kept = [c for c in state["index"].chunks if c.source != name]
    state["index"] = BM25Index(kept + new_chunks)
    state["index"].save(INDEX_PATH)
    return {"name": name, "chunks": len(new_chunks)}


@app.post("/api/ask")
def ask(req: AskRequest):
    hits = state["index"].search(req.question, k=req.k)
    passages = [
        {"source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 2), "text": h.chunk.text}
        for h in hits
    ]
    try:
        result = answer_question(req.question, hits)
    except (LLMNotConfigured, anthropic.AuthenticationError):
        return {"answer": None, "citations": [], "passages": passages,
                "error": "No valid ANTHROPIC_API_KEY set - showing the best matching passages only."}
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Rate limited by the model API, try again in a minute") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(503, "Could not reach the model API") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Model API error ({exc.status_code})") from exc
    return {
        "answer": result.text,
        "citations": [c.__dict__ for c in result.citations],
        "passages": passages,
        "error": None,
    }
