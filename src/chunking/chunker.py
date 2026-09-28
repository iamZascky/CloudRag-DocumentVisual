from typing import List, Dict

class DocumentChunker:
    """
    Handles chunking for visual document transcriptions, supporting page-level
    units and sliding window text chunking.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_page(self, text: str, page_metadata: Dict) -> List[Dict]:
        """
        Creates semantic chunks for a single document page while preserving metadata.
        """
        if not text:
            return []
            
        # For short pages/receipts, single page chunk is optimal for visual RAG
        if len(text) <= self.chunk_size:
            return [{
                "chunk_id": f"{page_metadata.get('file_path', 'doc')}_chunk_0",
                "text": text,
                "metadata": page_metadata
            }]
            
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "chunk_id": f"{page_metadata.get('file_path', 'doc')}_chunk_{idx}",
                "text": chunk_text,
                "metadata": page_metadata
            })
            start += self.chunk_size - self.overlap
            idx += 1
            
        return chunks
