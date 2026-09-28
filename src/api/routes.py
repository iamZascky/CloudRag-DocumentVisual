from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.retrieval.retriever import HybridRetriever
from src.llm.llm_client import QwenVisualReader
from src.utils.helpers import StructuredQAResponse
import time

app = FastAPI(
    title="Visual-RAG API",
    description="REST API for Visual-RAG with Hybrid Search (FTS5 + ChromaDB RRF) & Structured Output"
)

# Global singletons
_retriever = None
_reader = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def get_reader() -> QwenVisualReader:
    global _reader
    if _reader is None:
        _reader = QwenVisualReader()
    return _reader

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    mode: Optional[str] = "hybrid"

class SearchResult(BaseModel):
    final_rank: int
    rrf_score: float
    file_path: str
    doc_category: str
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    vector_similarity: Optional[float] = None
    highlight: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    structured: Optional[bool] = True

class APIAskResponse(BaseModel):
    question: str
    candidate_page: str
    doc_category: str
    rrf_score: float
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    data: StructuredQAResponse
    latency_seconds: float

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "Local Windows-Native Visual-RAG"}

@app.post("/search", response_model=List[SearchResult])
def search_endpoint(req: SearchRequest):
    retriever = get_retriever()
    results = retriever.retrieve(req.query, limit=req.limit)
    return results

@app.post("/ask", response_model=APIAskResponse)
def ask_endpoint(req: AskRequest):
    t0 = time.time()
    retriever = get_retriever()
    results = retriever.retrieve(req.question, limit=1)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found for question.")
    
    top = results[0]
    reader = get_reader()
    
    if req.structured:
        qa_data = reader.answer_question_structured(top["file_path"], req.question)
    else:
        raw_ans = reader.answer_question(top["file_path"], req.question, stream=False)
        qa_data = StructuredQAResponse(
            direct_answer=raw_ans,
            raw_answer=raw_ans
        )
        
    elapsed = time.time() - t0
    return APIAskResponse(
        question=req.question,
        candidate_page=top["file_path"],
        doc_category=top.get("doc_category", "DOCUMENT"),
        rrf_score=top["rrf_score"],
        fts_rank=top.get("fts_rank"),
        vector_rank=top.get("vector_rank"),
        data=qa_data,
        latency_seconds=round(elapsed, 2)
    )
