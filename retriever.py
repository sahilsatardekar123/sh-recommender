"""
retriever.py — FAISS vector index + semantic search over the SHL catalog.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "catalog.index"
META_PATH = "catalog_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once on first use
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_meta: list[dict] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_index_and_meta() -> tuple[faiss.Index, list[dict]]:
    global _index, _meta
    if _index is None or _meta is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{INDEX_PATH}'. "
                "Run `python build_index.py` first."
            )
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Catalog metadata not found at '{META_PATH}'. "
                "Run `python build_index.py` first."
            )
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    return _index, _meta


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings using the sentence transformer model."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    return embeddings.astype("float32")


def build_and_save_index(catalog: list[dict]) -> None:
    """
    Build a FAISS flat inner-product index from catalog items and save to disk.
    Each item is embedded as: "{name}. {description}"
    """
    texts = [
        f"{item['name']}. {item.get('description', '')}"
        for item in catalog
    ]
    print(f"Embedding {len(texts)} assessments...")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalised = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"✓ FAISS index saved to {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    print(f"✓ Metadata saved to {META_PATH}")


def search_catalog(query: str, k: int = 10) -> list[dict]:
    """
    Embed the query and return the top-k most semantically similar catalog items.

    Args:
        query: Natural language search string (e.g. "cognitive test for software engineers")
        k:     Number of results to return (default 10, max capped to index size)

    Returns:
        List of catalog dicts: [{name, url, test_type, description}, ...]
    """
    index, meta = _get_index_and_meta()

    k = min(k, index.ntotal)
    if k == 0:
        return []

    query_vec = embed_texts([query])  # shape (1, dim)
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(meta):
            results.append(meta[idx])
    return results


def get_all_catalog() -> list[dict]:
    """Return the full catalog metadata list (loaded from catalog_meta.json)."""
    _, meta = _get_index_and_meta()
    return meta


def get_item_by_name(name: str) -> dict | None:
    """
    Look up a catalog item by exact or partial name match.
    Returns the best match or None.
    """
    _, meta = _get_index_and_meta()
    name_lower = name.lower().strip()
    # Exact match first
    for item in meta:
        if item["name"].lower() == name_lower:
            return item
    # Partial match
    for item in meta:
        if name_lower in item["name"].lower():
            return item
    return None
