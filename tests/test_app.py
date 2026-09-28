import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import load_config, validate_extraction
from src.vectordb.sqlite_db import SqliteDocDatabase
from src.vectordb.vector_store import ChromaVectorStore
from src.retrieval.retriever import HybridRetriever

def test_config():
    cfg = load_config()
    assert cfg is not None
    assert "storage" in cfg
    assert "models" in cfg
    print("[PASS] test_config passed.")

def test_sqlite_db():
    db = SqliteDocDatabase()
    conn = db.get_connection()
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert count >= 0
    print(f"[PASS] test_sqlite_db passed ({count} docs).")

def test_chroma_vector_store():
    vs = ChromaVectorStore()
    count = vs.count()
    assert count >= 0
    print(f"[PASS] test_chroma_vector_store passed ({count} vectors).")

def test_hybrid_retriever():
    retriever = HybridRetriever()
    results = retriever.retrieve("bahan bakar bbm", limit=3)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    print(f"[PASS] test_hybrid_retriever passed (Top RRF: {results[0]['rrf_score']}).")

def test_validator():
    status = validate_extraction("RECEIPT", {
        "items": [{"price": 10000}, {"price": 15000}],
        "total_price": 25000
    })
    assert status == "VERIFIED"
    print("[PASS] test_validator passed.")

def test_structured_qa_schema():
    from src.utils.helpers import StructuredQAResponse, parse_first_numeric
    num = parse_first_numeric("Invoice Total: $ 12,450.00")
    assert num == 12450.0
    
    resp = StructuredQAResponse(
        direct_answer="Total expenditure is 12450.00",
        numeric_value=num,
        currency="USD",
        source_citation="Summary Report Table",
        breakdown=[{"label": "Hardware", "amount": 12450.0}],
        confidence="HIGH"
    )
    assert resp.numeric_value == 12450.0
    assert resp.currency == "USD"
    print("[PASS] test_structured_qa_schema passed.")

if __name__ == "__main__":
    print("=" * 50)
    print("  Running Visual-RAG Modular Test Suite")
    print("=" * 50)
    test_config()
    test_sqlite_db()
    test_chroma_vector_store()
    test_hybrid_retriever()
    test_validator()
    test_structured_qa_schema()
    print("=" * 50)
    print("  All tests passed successfully!")
    print("=" * 50)
