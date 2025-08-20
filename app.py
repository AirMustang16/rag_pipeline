from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import config
from embeddings import Embedder
from pinecone_store import ensure_index, delete_index, query
from ingest import ingest


app = FastAPI(title="RAG Service", version="0.1.0")


class IngestRequest(BaseModel):
    recreate_index: bool = False
    data_dir: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


def _llm_client():
    from openai import OpenAI
    return OpenAI()


def _rewrite_query(user_query: str, history: Optional[List[Dict[str, str]]]) -> str:
    client = _llm_client()
    sys = "Rewrite the user query to be best for semantic search over software product reviews. Keep it short; no punctuation."
    msgs = [{"role": "system", "content": sys}]
    if history:
        msgs += history[-4:]
    msgs.append({"role": "user", "content": user_query})
    try:
        out = client.chat.completions.create(model=config.OPENAI_MODEL, messages=msgs, temperature=0.2, max_tokens=32)
        return out.choices[0].message.content.strip()
    except Exception:
        return user_query


def _generate_answer(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = _llm_client()
    citations = []
    for i, m in enumerate(contexts, 1):
        meta = m["metadata"]
        citations.append({
            "id": m["id"],
            "rank": i,
            "title": meta.get("title"),
            "source_url": meta.get("source_url"),
            "source_domain": meta.get("source_domain"),
            "date": meta.get("date"),
            "snippet": meta.get("text"),
        })
    context_text = "\n\n".join([f"[{i}] {c['snippet']}" for i, c in enumerate(citations, 1)])
    sys = (
        "You are a helpful assistant answering questions using the provided review snippets. "
        "Cite sources as [1], [2], etc. Be concise and specific. If context is insufficient, say so."
    )
    prompt = (
        f"User question: {query}\n\n"
        f"Context snippets:\n{context_text}\n\n"
        "Answer the question using only the context above. Then provide 2-3 short follow-up suggestions."
    )
    out = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    answer = out.choices[0].message.content.strip()
    follow_ups = [
        "Summarize key pros and cons.",
        "Any trends by date or version?",
        "Provide quotes supporting the answer.",
    ]
    return {"answer": answer, "citations": citations, "follow_ups": follow_ups}


@app.get("/health")
def health():
    ok = bool(config.PINECONE_API_KEY)
    return {"status": "ok" if ok else "missing_keys", "pinecone_index": config.PINECONE_INDEX}


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    if not config.PINECONE_API_KEY:
        raise HTTPException(status_code=400, detail="PINECONE_API_KEY is required")
    if req.recreate_index:
        delete_index()
    dim = Embedder().dim
    ensure_index(dim)
    stats = ingest(recreate_index=False, data_dir=req.data_dir)
    return {"ok": True, **stats, "index": config.PINECONE_INDEX, "namespace": config.PINECONE_NAMESPACE}


@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not config.PINECONE_API_KEY:
        raise HTTPException(status_code=400, detail="PINECONE_API_KEY is required")
    embedder = Embedder()
    ensure_index(embedder.dim)
    search_text = _rewrite_query(req.query, req.history)
    qvec = embedder.embed([search_text])[0].tolist()
    res = query(qvec, top_k=req.top_k, filter=req.filter)
    matches = res.matches or []
    contexts = [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in matches]
    return _generate_answer(req.query, contexts)



