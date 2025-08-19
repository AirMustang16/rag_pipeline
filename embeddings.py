from typing import List
import numpy as np
import config


def _sbert_loader():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.SBERT_MODEL)


class Embedder:
    def __init__(self):
        self.kind = "sbert"
        if config.EMBED_PROVIDER in ("openai", "auto") and config.OPENAI_API_KEY:
            self.kind = "openai"
        elif config.EMBED_PROVIDER == "sbert":
            self.kind = "sbert"
        self._model = None

    @property
    def dim(self) -> int:
        if self.kind == "openai":
            return 1536
        return 384  # all-MiniLM-L6-v2

    def _ensure_model(self):
        if self.kind == "sbert" and self._model is None:
            self._model = _sbert_loader()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self.kind == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
            vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
            return np.vstack(vecs)
        # SBERT
        self._ensure_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")



