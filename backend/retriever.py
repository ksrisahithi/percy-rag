"""
Loads the FAISS index built by embedder.py and retrieves the top-k most
relevant chunks for a query string.
"""

import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


@lru_cache(maxsize=1)
def _load_resources():
    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("FAISS index not found. Run embedder.py first.")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        store = pickle.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    return index, store["texts"], store["metadata"], model


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Returns top_k chunks as dicts with keys: text, source, score."""
    index, texts, metadata, model = _load_resources()

    embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embedding)

    scores, indices = index.search(embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": texts[idx],
            "source": metadata[idx]["source"],
            "score": float(score),
        })
    return results


def format_context(results: list[dict]) -> str:
    """Joins retrieved chunks into a single context block for the LLM prompt."""
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source'].replace('_', ' ')}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    query = input("Test query: ")
    hits = retrieve(query)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] {h['source']}  score={h['score']:.3f}")
        print(h["text"][:300])
