from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import config


def client() -> Pinecone:
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index(dim: int):
    pc = client()
    if config.PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    return pc.Index(config.PINECONE_INDEX)


def delete_index():
    pc = client()
    if config.PINECONE_INDEX in [i.name for i in pc.list_indexes()]:
        pc.delete_index(config.PINECONE_INDEX)


def upsert(vectors: List[Dict[str, Any]]):
    index = client().Index(config.PINECONE_INDEX)
    index.upsert(vectors=vectors, namespace=config.PINECONE_NAMESPACE)


def query(vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None):
    index = client().Index(config.PINECONE_INDEX)
    return index.query(
        vector=vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        namespace=config.PINECONE_NAMESPACE,
        filter=filter or None,
    )



