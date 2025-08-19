import hashlib
import json
from typing import Dict, List, Iterable, Optional, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import config
from embeddings import Embedder
from pinecone_store import ensure_index, upsert


def _read_rows(data_dir: Optional[Union[str, Path]] = None) -> Iterable[Dict]:
    base_dir = Path(data_dir) if data_dir else Path(config.RAG_DATA_DIR)
    print(f"[ingest] Scanning data directory: {base_dir}")
    if not base_dir.exists():
        print(f"[ingest] Directory does not exist: {base_dir}")
        return

    patterns = ["*_clean.jsonl", "*.jsonl", "*.json", "*_clean.csv"]
    files: List[Path] = []
    for pat in patterns:
        matched = sorted(base_dir.glob(pat))
        if matched:
            print(f"[ingest] Found {len(matched)} files for pattern '{pat}'")
        files.extend(matched)
    # De-duplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for p in files:
        if p not in seen:
            unique_files.append(p)
            seen.add(p)
    if not unique_files:
        print("[ingest] No data files found.")
        return

    total_yielded = 0
    for p in unique_files:
        try:
            if p.suffix.lower() == ".jsonl" or p.name.endswith(".jsonl"):
                print(f"[ingest] Reading JSONL: {p}")
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            total_yielded += 1
                            yield obj
                        except Exception:
                            continue
                continue

            if p.suffix.lower() == ".json":
                print(f"[ingest] Reading JSON: {p}")
                with p.open("r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for obj in data:
                                if isinstance(obj, dict):
                                    total_yielded += 1
                                    yield obj
                        elif isinstance(data, dict):
                            total_yielded += 1
                            yield data
                        else:
                            print(f"[ingest] Unsupported JSON top-level type in {p}: {type(data)}")
                    except Exception as e:
                        print(f"[ingest] Failed to read JSON {p}: {e}")
                continue

            if p.suffix.lower() == ".csv":
                print(f"[ingest] Reading CSV: {p}")
                try:
                    df = pd.read_csv(p, dtype=str, keep_default_na=False)
                    for _, row in df.iterrows():
                        total_yielded += 1
                        yield row.to_dict()
                except Exception as e:
                    print(f"[ingest] Failed to read CSV {p}: {e}")
                continue
        finally:
            pass
    print(f"[ingest] Total records discovered: {total_yielded}")


def _as_text(value) -> str:
    """Convert a value to a safe text string; coerce NaN/None to empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _chunk(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def _doc_id(row: Dict) -> str:
    # Normalize potential NaN/None values to strings for stable hashing
    title_str = _as_text(row.get('title'))
    body_str = _as_text(row.get('body'))
    author_str = _as_text(row.get('author'))
    date_str = _as_text(row.get('date'))
    source_domain_str = _as_text(row.get('source_domain'))
    base = f"{source_domain_str}|{date_str}|{author_str}|{title_str[:60]}|{body_str[:120]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_payloads(data_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
    payloads: List[Dict] = []
    for row in _read_rows(data_dir=data_dir):
        body = _as_text(row.get("body"))
        title = _as_text(row.get("title"))
        pros = _as_text(row.get("pros"))
        cons = _as_text(row.get("cons"))
        text = "\n\n".join([t for t in [title, body, f"Pros: {pros}" if pros else "", f"Cons: {cons}" if cons else ""] if t])
        chunks = _chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            continue
        base_id = _doc_id(row)
        # Sanitize metadata to ensure JSON-serializable values (no NaN)
        rating = row.get("rating")
        try:
            rating = None if pd.isna(rating) else rating
        except Exception:
            pass
        # If no numeric rating but rating_raw exists, try to parse a float
        if rating in (None, ""):
            rating_raw = _as_text(row.get("rating_raw"))
            try:
                if rating_raw:
                    rating = float(rating_raw.split("/")[0])
            except Exception:
                rating = None
        meta_common = {
            "title": title or None,
            "author": _as_text(row.get("author")) or None,
            "date": _as_text(row.get("date")) or None,
            "rating": rating,
            "source_url": _as_text(row.get("source_url")) or None,
            "source_domain": _as_text(row.get("source_domain")) or None,
        }
        # Remove None/null metadata entries (Pinecone does not accept null values)
        meta_clean = {k: v for k, v in meta_common.items() if v is not None and (not isinstance(v, str) or v.strip() != "")}
        for i, ch in enumerate(chunks):
            payloads.append({
                "id": f"{base_id}#{i}",
                "values": None,  # fill later
                "metadata": {**meta_clean, "text": ch, "chunk_index": i},
            })
    print(f"[ingest] Built {len(payloads)} chunk payloads")
    return payloads


def ingest(recreate_index: bool = False, data_dir: Optional[Union[str, Path]] = None):
    embedder = Embedder()
    ensure_index(embedder.dim)

    payloads = build_payloads(data_dir=data_dir)
    if not payloads:
        return {"ingested": 0, "batches": 0}

    texts = [p["metadata"]["text"] for p in payloads]
    vecs = embedder.embed(texts)

    for p, v in zip(payloads, vecs):
        p["values"] = v.tolist()

    total = 0
    for i in tqdm(range(0, len(payloads), config.BATCH_SIZE), desc="Upserting"):
        batch = payloads[i:i + config.BATCH_SIZE]
        upsert(batch)
        total += len(batch)

    return {"ingested": total, "batches": (len(payloads) + config.BATCH_SIZE - 1) // config.BATCH_SIZE}



