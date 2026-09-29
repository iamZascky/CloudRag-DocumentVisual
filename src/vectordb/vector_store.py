import os
import sys
import json
import math
import sqlite3
from typing import List, Dict, Optional, Set
from src.embeddings.embedder import MultilingualE5Embedder
from src.utils.helpers import load_config

config = load_config()
STORAGE_DIR = os.path.abspath(config.get("storage", {}).get("vector_db", "./storage/vectors"))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors in pure Python."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class PureVectorStore:
    """
    Lightweight, 100% thread-safe Vector Store using SQLite and pure Python Cosine Similarity.
    Zero C++ threading (no ChromaDB/HNSW thread explosion).
    Perfect for CloudLinux Shared Hosting (nproc strictly 1).
    """
    def __init__(self, db_path: Optional[str] = None, embedder: Optional[MultilingualE5Embedder] = None):
        if db_path is None:
            os.makedirs(STORAGE_DIR, exist_ok=True)
            self.db_path = os.path.join(STORAGE_DIR, "vectors_pure.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.embedder = embedder or MultilingualE5Embedder()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS page_vectors (
            id TEXT PRIMARY KEY,
            file_path TEXT UNIQUE,
            doc_category TEXT,
            title TEXT,
            snippet TEXT,
            embedding_json TEXT
        );
        """)
        conn.commit()
        conn.close()

    def upsert_page(self, file_path: str, text: str, category: str, title: str):
        norm_path = os.path.normpath(file_path)
        content_to_embed = f"{title}\n{text}".strip()
        embedding = self.embedder.embed_passages([content_to_embed])[0]
        snippet = text[:1500] if text else ""
        embedding_str = json.dumps(embedding)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO page_vectors (id, file_path, doc_category, title, snippet, embedding_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            file_path=excluded.file_path,
            doc_category=excluded.doc_category,
            title=excluded.title,
            snippet=excluded.snippet,
            embedding_json=excluded.embedding_json;
        """, (norm_path, norm_path, category or "OTHER", title or "Unknown Title", snippet, embedding_str))
        conn.commit()
        conn.close()

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, doc_category, title, snippet, embedding_json FROM page_vectors")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_embedding = self.embedder.embed_query(query)
        scored_results = []

        for row in rows:
            try:
                emb = json.loads(row["embedding_json"])
                sim = cosine_similarity(query_embedding, emb)
                scored_results.append({
                    "file_path": row["file_path"],
                    "doc_category": row["doc_category"] or "OTHER",
                    "title": row["title"] or "",
                    "similarity": round(max(0.0, sim), 4),
                    "snippet": (row["snippet"] or "")[:200]
                })
            except Exception:
                continue

        # Sort descending by similarity
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:limit]

    def count(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM page_vectors")
        cnt = cursor.fetchone()[0]
        conn.close()
        return cnt

    def get_indexed_paths(self) -> Set[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM page_vectors")
        paths = {row[0] for row in cursor.fetchall()}
        conn.close()
        return paths

    def delete_page(self, file_path: str):
        norm_path = os.path.normpath(file_path)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM page_vectors WHERE id = ?", (norm_path,))
        conn.commit()
        conn.close()

# Backward-compatibility alias
ChromaVectorStore = PureVectorStore
