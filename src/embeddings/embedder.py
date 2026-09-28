import os
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.utils.helpers import load_config

config = load_config()
storage_cfg = config.get("storage", {})
embed_cfg = config.get("models", {}).get("embedding", {})

os.environ["HF_HOME"] = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface"))

MODEL_NAME = embed_cfg.get("model_name", "intfloat/multilingual-e5-small")
DEVICE = embed_cfg.get("device", "cpu")
NORMALIZE = embed_cfg.get("normalize", True)

class MultilingualE5Embedder:
    """
    Manages embedding generation using intfloat/multilingual-e5-small on CPU.
    Automatically applies required task prefixes ('passage: ' and 'query: ').
    """
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.cache_folder = os.path.abspath(storage_cfg.get("hf_cache_dir", "./storage/models/huggingface/hub"))
        print(f"Loading embedder {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_folder)

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embeds document texts for indexing (prepends 'passage: ')."""
        prefixed = [f"passage: {t.strip()}" for t in texts]
        embeddings = self.model.encode(prefixed, normalize_embeddings=NORMALIZE)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query (prepends 'query: ')."""
        prefixed = f"query: {query.strip()}"
        embedding = self.model.encode([prefixed], normalize_embeddings=NORMALIZE)[0]
        return embedding.tolist()
