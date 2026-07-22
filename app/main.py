import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ingestion import extract_text, chunk_text
from app.knowledge_graph import store

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Industrial Knowledge Intelligence - Ingestion & Knowledge Graph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    doc_id = f"DOC-{uuid.uuid4().hex[:8]}"
    dest = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text(str(dest))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in document (even after OCR).")

    chunks = chunk_text(text)
    result = store.ingest_document(
        doc_id=doc_id,
        filename=file.filename,
        text=text,
        chunks=chunks,
        upload_date=datetime.utcnow().isoformat(),
    )
    return result


@app.post("/api/ingest_samples")
def ingest_samples():
    """Load the bundled sample industrial documents for demo purposes."""
    results = []
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        doc_id = f"DOC-{uuid.uuid4().hex[:8]}"
        text = extract_text(str(path))
        chunks = chunk_text(text)
        result = store.ingest_document(
            doc_id=doc_id,
            filename=path.name,
            text=text,
            chunks=chunks,
            upload_date=datetime.utcnow().isoformat(),
        )
        results.append(result)
    return {"ingested": results}


@app.get("/api/graph")
def get_graph(entity_type: str | None = None, search: str | None = None):
    return store.get_graph_json(entity_type_filter=entity_type, search=search)


@app.get("/api/entity/{key:path}")
def get_entity(key: str):
    detail = store.get_entity_detail(key)
    if not detail:
        raise HTTPException(status_code=404, detail="Entity not found")
    return detail


@app.get("/api/search")
def search(q: str):
    return {"query": q, "results": store.search_documents(q)}


@app.get("/api/documents")
def list_documents():
    return [
        {"doc_id": doc_id, "filename": d["filename"], "upload_date": d["upload_date"],
         "metadata": d["metadata"], "entity_count": d["entity_count"]}
        for doc_id, d in store.documents.items()
    ]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    if doc_id not in store.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return store.documents[doc_id]


@app.get("/api/stats")
def stats():
    return store.stats()


@app.post("/api/reset")
def reset():
    store.reset()
    return {"status": "reset"}
