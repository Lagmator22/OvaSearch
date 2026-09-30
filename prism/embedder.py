"""CPU text/code embedder used by the index and search paths."""

from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL = os.environ.get("PRISM_MODEL", "nomic-ai/CodeRankEmbed")

# Query prefixes from the model cards. Documents get no prefix for these models.
QUERY_PREFIX = {
    "nomic-ai/CodeRankEmbed": "Represent this query for searching relevant code: ",
    "Salesforce/SFR-Embedding-Code-400M_R":
        "Instruct: Given Code or Text, retrieval relevant content\nQuery: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_len: int = 1024,
                 batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        self.model.max_seq_length = max_len
        self.backend = "sentence-transformers (PyTorch CPU)"
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 64).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        q = QUERY_PREFIX.get(self.model_name, "") + query
        return self.model.encode([q], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
