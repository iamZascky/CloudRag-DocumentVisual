import os
from typing import List, Dict, Optional, Set
import chromadb
from src.embeddings.embedder import MultilingualE5Embedder
from src.utils.helpers import load_config

config = load_config()
CHROMA_DIR = os.path.abspath(config.get("storage", {}).get("chroma_db", "./storage/chroma"))

class ChromaVectorStore:
    """
    Manages vector database operations using ChromaDB.
    """
    def __init__(self, persist_dir: str = CHROMA_DIR, embedder: Optional[MultilingualE5Embedder] = None):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="doc_pages_e5",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder or MultilingualE5Embedder()

    def upsert_page(self, file_path: str, text: str, category: str, title: str):
        norm_path = os.path.normpath(file_path)
        content_to_embed = f"{title}\n{text}".strip()
        embedding = self.embedder.embed_passages([content_to_embed])[0]
        snippet = text[:1500] if text else ""
        
        self.collection.upsert(
            ids=[norm_path],
            embeddings=[embedding],
            documents=[snippet],
            metadatas=[{
                "file_path": norm_path,
                "doc_category": category or "OTHER",
                "title": title or "Unknown Title"
            }]
        )

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []
            
        query_embedding = self.embedder.embed_query(query)
        n_results = min(limit, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances", "documents"]
        )
        
        output = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            documents = results["documents"][0] if results.get("documents") else []
            
            for idx, doc_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                dist = distances[idx] if idx < len(distances) else 1.0
                doc_text = documents[idx] if idx < len(documents) else ""
                similarity = max(0.0, 1.0 - dist)
                
                output.append({
                    "file_path": meta.get("file_path", doc_id),
                    "doc_category": meta.get("doc_category", "OTHER"),
                    "title": meta.get("title", ""),
                    "similarity": round(similarity, 4),
                    "snippet": doc_text[:200]
                })
        return output

    def count(self) -> int:
        return self.collection.count()

    def get_indexed_paths(self) -> Set[str]:
        data = self.collection.get(include=[])
        if data and "ids" in data:
            return set(data["ids"])
        return set()

    def delete_page(self, file_path: str):
        norm_path = os.path.normpath(file_path)
        self.collection.delete(ids=[norm_path])
