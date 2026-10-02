import os
from typing import List, Dict, Optional
from src.vectordb.sqlite_db import SqliteDocDatabase
from src.vectordb.vector_store import ChromaVectorStore
from src.utils.helpers import load_config, expand_general_document_query

config = load_config()
retrieval_cfg = config.get("retrieval", {})
DEFAULT_K = retrieval_cfg.get("rrf_k", 60)
DEFAULT_DEPTH = retrieval_cfg.get("search_depth", 15)

class HybridRetriever:
    """
    Orchestrates Hybrid Retrieval by fusing SQLite FTS5 (Lexical)
    and ChromaDB (Semantic) using Reciprocal Rank Fusion (RRF)
    with universal query expansion for general documents.
    """
    def __init__(
        self,
        sqlite_db: Optional[SqliteDocDatabase] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        k: int = DEFAULT_K
    ):
        self.sqlite_db = sqlite_db or SqliteDocDatabase()
        self.vector_store = vector_store or ChromaVectorStore()
        self.k = k

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_depth: int = DEFAULT_DEPTH
    ) -> List[Dict]:
        """
        Executes hybrid search with query expansion and returns candidates ranked by RRF score.
        """
        # Apply lightweight pure-Python universal query expansion
        expanded_query = expand_general_document_query(query)
        sanitized = expanded_query.replace('"', '').replace("'", "").strip()
        fts_results = []
        
        # 1. Try exact phrase match first in FTS (using original query to prioritize exact hits)
        orig_sanitized = query.replace('"', '').replace("'", "").strip()
        try:
            fts_results = self.sqlite_db.search_text(f'"{orig_sanitized}"', limit=search_depth)
        except Exception:
            pass
            
        # 2. Fall back to word prefix matching in FTS with expanded terms
        if not fts_results:
            tokens = [w for w in sanitized.replace("?", "").replace("!", "").replace(",", "").split() if len(w) > 1]
            if tokens:
                fts_query = " OR ".join(f'"{t}"*' for t in tokens[:12])
                try:
                    fts_results = self.sqlite_db.search_text(fts_query, limit=search_depth)
                except Exception:
                    pass
                    
        # 3. Vector semantic search using expanded query
        vector_results = []
        # Fast path: If FTS returned sufficient direct lexical matches (>= limit), we already have the right pages!
        # This keeps query retrieval instantaneous (< 0.01s) instead of waiting for slow CPU embeddings.
        if len(fts_results) < limit:
            try:
                vector_results = self.vector_store.search(expanded_query, limit=search_depth)
            except Exception as v_err:
                print(f"[HybridRetriever] Vector search skipped/failed: {v_err}")
        
        # 4. Compute RRF scores
        rrf_scores = {}
        doc_registry = {}
        
        # Process FTS ranks
        for rank, item in enumerate(fts_results, start=1):
            path = os.path.normpath(item["file_path"])
            rrf_scores[path] = rrf_scores.get(path, 0.0) + (1.0 / (self.k + rank))
            doc_registry[path] = {
                "file_path": path,
                "doc_category": item.get("doc_category", "OTHER"),
                "highlight": item.get("highlight", ""),
                "fts_rank": rank,
                "vector_rank": None,
                "vector_similarity": None
            }
            
        # Process Vector ranks
        for rank, item in enumerate(vector_results, start=1):
            path = os.path.normpath(item["file_path"])
            rrf_scores[path] = rrf_scores.get(path, 0.0) + (1.0 / (self.k + rank))
            
            if path not in doc_registry:
                doc_registry[path] = {
                    "file_path": path,
                    "doc_category": item.get("doc_category", "OTHER"),
                    "highlight": item.get("snippet", ""),
                    "fts_rank": None,
                    "vector_rank": rank,
                    "vector_similarity": item.get("similarity")
                }
            else:
                doc_registry[path]["vector_rank"] = rank
                doc_registry[path]["vector_similarity"] = item.get("similarity")
                if not doc_registry[path]["highlight"]:
                    doc_registry[path]["highlight"] = item.get("snippet", "")
                    
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for rank_idx, (path, score) in enumerate(sorted_docs[:limit], start=1):
            entry = doc_registry[path].copy()
            entry["final_rank"] = rank_idx
            entry["rrf_score"] = round(score, 5)
            final_results.append(entry)
            
        return final_results
