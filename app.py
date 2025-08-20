from typing import List, Optional, Dict, Any
import json
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
    sys = (
        "Rewrite the user query for the best semantic search over customer reviews of the 'Reviews for Jira' app. "
        "Output a short, search-oriented phrase with no punctuation or quotes. "
        "Preserve key entities and filters if present (features, versions, authors, sources, dates, ratings). "
        "Do not add or infer information; do not change the meaning."
    )
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
        "You are a product expert answering questions about the 'Reviews for Jira' app using ONLY the provided user review snippets. "
        "Cite sources with bracket numbers [1], [2], etc., matching the snippet indices shown in the context. "
        "Be concise, specific, and factual; prefer concrete details (ratings, dates, sources). "
        "If the context is insufficient to answer, say so explicitly and avoid speculation."
    )
    prompt = (
        f"User question: {query}\n\n"
        f"Context snippets:\n{context_text}\n\n"
        "Answer directly using only the context above. Cite claims with [n] where n is the snippet number. "

    )
    out = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    answer = out.choices[0].message.content.strip()
    follow_ups = [
        "Fetch the top rated review from www.softwareadvice.com.",
        "Whats are the pros from Elena V.",
        "What is Mrudul P's concern about Jira?",
    ]
    return {"answer": answer, "citations": citations, "follow_ups": follow_ups}


def _sanitize_pinecone_filter(raw: Any) -> Optional[Dict[str, Any]]:
    """Validate and reduce an arbitrary JSON object into a Pinecone-safe metadata filter.

    Allowed fields: author (str), source_domain (str), date (str), rating (number).
    Allowed ops:
      - Strings: $eq, $in
      - Numbers (rating): $eq, $gt, $gte, $lt, $lte, $in
      - Dates (as strings): $eq, $gte, $lte
    Returns None if nothing valid remains.
    """
    if not isinstance(raw, dict):
        return None

    allowed_fields = {
        "author": "str",
        "source_domain": "str",
        "date": "str",
        "rating": "num",
    }
    string_ops = {"$eq", "$in"}
    number_ops = {"$eq", "$gt", "$gte", "$lt", "$lte", "$in"}
    date_ops = {"$eq", "$gte", "$lte"}

    sanitized: Dict[str, Any] = {}

    for field, constraint in raw.items():
        if field not in allowed_fields:
            continue
        expected_type = allowed_fields[field]

        # Allow shorthand: { field: value } -> { field: { "$eq": value } }
        if not isinstance(constraint, dict):
            constraint = {"$eq": constraint}

        if not isinstance(constraint, dict):
            continue

        clean_ops: Dict[str, Any] = {}
        ops_allowed = date_ops if field == "date" else (number_ops if field == "rating" else string_ops)

        for op, value in constraint.items():
            if op not in ops_allowed:
                continue
            if expected_type == "str":
                if op == "$in":
                    if isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value):
                        clean_ops[op] = value
                else:
                    if isinstance(value, str) and value.strip():
                        clean_ops[op] = value
            elif expected_type == "num":
                if op == "$in":
                    if isinstance(value, list):
                        nums = []
                        for v in value:
                            if isinstance(v, (int, float)):
                                nums.append(float(v))
                        if nums:
                            clean_ops[op] = nums
                else:
                    if isinstance(value, (int, float)):
                        clean_ops[op] = float(value)
            else:  # date as string
                if isinstance(value, str) and value.strip():
                    clean_ops[op] = value

        if clean_ops:
            sanitized[field] = clean_ops

    return sanitized or None


def _extract_filter(user_query: str) -> Optional[Dict[str, Any]]:
    """Use the LLM to infer a Pinecone metadata filter from a natural-language query.

    Returns a dict suitable for Pinecone's `filter` parameter or None.
    """
    client = _llm_client()
    sys = (
        "Extract Pinecone metadata filters from natural language queries about the 'Reviews for Jira' app. "
        "Only output strict JSON with a top-level key 'filter'. "
        "Supported fields: author (string), source_domain (string), date (YYYY-MM-DD string), rating (number). "
        "Supported operators: $eq, $in for strings; $eq, $gt, $gte, $lt, $lte, $in for rating; $eq, $gte, $lte for date. "
        "If no filter is implied, set filter to null. Do not include unsupported fields or operators."
    )
    # Few-shot examples to steer consistent structure
    examples = [
        {
            "q": "What's the review added by Jose?",
            "f": {"author": {"$eq": "Jose"}},
        },
        {
            "q": "Show reviews from softwareadvice.com in 2024",
            "f": {"source_domain": {"$eq": "softwareadvice.com"}, "date": {"$gte": "2024-01-01", "$lte": "2024-12-31"}},
        },
        {
            "q": "What are users rating it above 4?",
            "f": {"rating": {"$gt": 4}},
        },
        {
            "q": "Summarize pros and cons",
            "f": None,
        },
    ]
    instruction = {
        "examples": examples,
        "output_schema": {
            "filter": "object|null; Pinecone metadata filter built only from allowed fields and operators",
        },
    }
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps({"instruction": instruction, "query": user_query})},
    ]
    try:
        out = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = out.choices[0].message.content
        obj = json.loads(content)
        filt = obj.get("filter")
        return _sanitize_pinecone_filter(filt)
    except Exception:
        return None


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
    auto_filter = _extract_filter(req.query)
    effective_filter = req.filter if req.filter is not None else auto_filter
    print(effective_filter)
    qvec = embedder.embed([search_text])[0].tolist()
    res = query(qvec, top_k=req.top_k, filter=effective_filter)
    matches = res.matches or []
    contexts = [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in matches]
    return _generate_answer(req.query, contexts)



