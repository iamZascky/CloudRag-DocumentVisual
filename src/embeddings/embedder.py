import os
from typing import List, Union
from src.utils.helpers import load_config

config = load_config()
storage_cfg = config.get("storage", {})
embed_cfg = config.get("models", {}).get("embedding", {})

os.environ["HF_HOME"] = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface"))

MODEL_NAME = embed_cfg.get("model_name", "intfloat/multilingual-e5-small")
DEVICE = embed_cfg.get("device", "cpu")
NORMALIZE = embed_cfg.get("normalize", True)

class GeminiCloudEmbedder:
    """
    Lightweight Cloud Embedder using Google Gemini Embedding REST API.
    Zero PyTorch, Zero sentence-transformers, Zero RAM footprint (< 2 MB).
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

    def _call_api(self, text: str) -> List[float]:
        import requests
        if not self.api_key:
            return [0.0] * 768
        url = f"{self.base_url}?key={self.api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text[:2000]}]}
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("embedding", {}).get("values", [0.0] * 768)
        except Exception:
            pass
        return [0.0] * 768

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return [self._call_api(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._call_api(query)

class MultilingualE5Embedder:
    """
    Adaptive Embedder:
    Uses GeminiCloudEmbedder if AI_MODE=cloud or on shared hosting.
    Uses SentenceTransformer E5 on local machines with dedicated resources.
    """
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        ai_mode = os.getenv("AI_MODE", "").lower()
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        
        # If in cloud mode or torch is absent, use GeminiCloudEmbedder
        if ai_mode == "cloud" or gemini_key:
            print("[Embedder] ☁️ Using lightweight GeminiCloudEmbedder (Zero torch/RAM)...")
            self._backend = GeminiCloudEmbedder(api_key=gemini_key)
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.model_name = model_name
                self.device = device
                self.cache_folder = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface/hub"))
                print(f"[Embedder] Loading local embedder {self.model_name} on {self.device}...")
                self._model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_folder)
                self._backend = None
            except Exception as e:
                print(f"[Embedder] Local SentenceTransformer unavailable ({e}). Falling back to GeminiCloudEmbedder...")
                self._backend = GeminiCloudEmbedder(api_key=gemini_key)

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        if self._backend:
            return self._backend.embed_passages(texts)
        prefixed = [f"passage: {t.strip()}" for t in texts]
        embeddings = self._model.encode(prefixed, normalize_embeddings=NORMALIZE)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        if self._backend:
            return self._backend.embed_query(query)
        prefixed = f"query: {query.strip()}"
        embedding = self._model.encode([prefixed], normalize_embeddings=NORMALIZE)[0]
        return embedding.tolist()
