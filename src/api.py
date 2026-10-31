import json
import logging
import os
import re
from typing import Optional

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
_PIPELINE_REGISTRY: dict[str, RAGPipeline] = {}
DATA_DIR = os.path.normpath(os.path.abspath("data"))


def get_safe_data_path(user_input_path: str) -> str:
    """
    Validates user-provided document paths to eliminate path traversal vulnerabilities (CWE-22 / CWE-73).
    Extracts the basename and matches it strictly against verified files present in DATA_DIR.
    """
    clean_name = os.path.basename(user_input_path.strip())
    if not clean_name or not clean_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid document path. Only PDF files located in data/ are accessible.",
        )

    if os.path.isdir(DATA_DIR):
        for existing_file in os.listdir(DATA_DIR):
            if existing_file == clean_name:
                verified_path = os.path.normpath(os.path.join(DATA_DIR, existing_file))
                if verified_path.startswith(DATA_DIR):
                    return verified_path

    raise HTTPException(
        status_code=404,
        detail=f"Document '{clean_name}' not found in data directory.",
    )


class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User query about the document")
    doc_id: str | None = Field(None, description="Document ID returned during upload")
    document_path: str | None = Field(None, description="Path or filename of PDF inside data/")
    chat_history: list[Message] = Field(default_factory=list, description="Prior conversation turns")


class SourceItem(BaseModel):
    file: str
    page: int
    score: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    doc_id: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    path: str
    pages: int
    chunks: int


def _get_or_create_pipeline(doc_id: str | None, doc_path: str | None) -> RAGPipeline:
    if doc_id and doc_id in _PIPELINE_REGISTRY:
        return _PIPELINE_REGISTRY[doc_id]

    if not doc_path and doc_id:
        for d_id, pipe in _PIPELINE_REGISTRY.items():
            if d_id == doc_id:
                return pipe

    if not doc_path:
        raise HTTPException(
            status_code=400,
            detail="Document not found. Please provide a valid doc_id or document_path.",
        )

    safe_target_path = get_safe_data_path(doc_path)
    pipeline = RAGPipeline(safe_target_path)
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


@app.get("/documents", response_model=list[DocumentInfo])
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
    raw_filename = file.filename or "upload.pdf"
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", os.path.basename(raw_filename))
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(DATA_DIR, exist_ok=True)
    destination_path = os.path.normpath(os.path.join(DATA_DIR, clean_name))
    if not destination_path.startswith(DATA_DIR):
        raise HTTPException(status_code=400, detail="Security error: Invalid target destination.")

    contents = await file.read()
    with open(destination_path, "wb") as f:
        f.write(contents)

    try:
        pipeline = RAGPipeline(destination_path)
        _PIPELINE_REGISTRY[pipeline.doc_id] = pipeline
        return DocumentInfo(
            doc_id=pipeline.doc_id,
            filename=clean_name,
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
