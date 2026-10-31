import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production Hybrid RAG Engine API",
    description="REST & SSE Streaming API for Document Grounded Hybrid Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory pipeline registry: doc_id -> RAGPipeline
_PIPELINE_REGISTRY: Dict[str, RAGPipeline] = {}


class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User query about the document")
    doc_id: Optional[str] = Field(None, description="Document ID returned during upload")
    document_path: Optional[str] = Field(None, description="Path to PDF if doc_id not provided")
    chat_history: List[Message] = Field(default_factory=list, description="Prior conversation turns")


class SourceItem(BaseModel):
    file: str
    page: int
    score: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    doc_id: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    path: str
    pages: int
    chunks: int


def _get_or_create_pipeline(doc_id: Optional[str], doc_path: Optional[str]) -> RAGPipeline:
    if doc_id and doc_id in _PIPELINE_REGISTRY:
        return _PIPELINE_REGISTRY[doc_id]

    target_path = doc_path
    if not target_path and doc_id:
        # Search registry for matching doc_id
        for d_id, pipe in _PIPELINE_REGISTRY.items():
            if d_id == doc_id:
                return pipe

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document not found or not indexed. Please upload document first.",
        )

    pipeline = RAGPipeline(target_path)
    _PIPELINE_REGISTRY[pipeline.doc_id] = pipeline
    return pipeline


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chroma_dir": settings.CHROMA_DB_DIR,
        "loaded_documents": len(_PIPELINE_REGISTRY),
    }


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    docs = []
    for doc_id, pipe in _PIPELINE_REGISTRY.items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=os.path.basename(pipe.document_path),
                path=pipe.document_path,
                pages=pipe.metrics.get("document_pages", 0),
                chunks=pipe.metrics.get("chunks_created", 0),
            )
        )
    return docs


@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    # Sanitize filename
    safe_filename = Path(file.filename or "upload.pdf").name
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("data", exist_ok=True)
    destination_path = os.path.join("data", safe_filename)

    contents = await file.read()
    with open(destination_path, "wb") as f:
        f.write(contents)

    try:
        pipeline = RAGPipeline(destination_path)
        _PIPELINE_REGISTRY[pipeline.doc_id] = pipeline
        return DocumentInfo(
            doc_id=pipeline.doc_id,
            filename=safe_filename,
            path=destination_path,
            pages=pipeline.metrics.get("document_pages", 0),
            chunks=pipeline.metrics.get("chunks_created", 0),
        )
    except Exception as exc:
        logger.exception("Failed to process uploaded PDF")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    pipeline = _get_or_create_pipeline(req.doc_id, req.document_path)
    history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history]

    res = pipeline.generate_answer(req.question, history_dicts)
    sources = [
        SourceItem(
            file=s["file"],
            page=s["page"],
            score=s["score"],
            text=s["text"],
        )
        for s in res["sources"]
    ]

    return QueryResponse(
        question=req.question,
        answer=res["answer"],
        sources=sources,
        doc_id=pipeline.doc_id,
    )


@app.post("/stream")
def stream_document_query(req: QueryRequest):
    pipeline = _get_or_create_pipeline(req.doc_id, req.document_path)
    history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history]

    retrieved_docs = pipeline.retrieve_chunks(req.question, history_dicts)

    def event_generator():
        for token in pipeline.stream_answer(req.question, retrieved_docs, history_dicts):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

