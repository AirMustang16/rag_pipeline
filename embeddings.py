from typing import List
import numpy as np
import config


class Embedder:
    def __init__(self):
        pass

    @property
    def dim(self) -> int:
        # OpenAI text-embedding-3-small
        return 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
        vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        return np.vstack(vecs)



