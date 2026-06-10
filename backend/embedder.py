"""
Loads raw .txt files from data/raw/, splits them into overlapping chunks,
generates embeddings with sentence-transformers, and saves a FAISS index
alongside a pickle of chunk texts and source metadata.

Run once after scraper.py:
    python backend/embedder.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 100    # characters


def load_raw_files(raw_dir: Path) -> list[dict]:
    docs = []
    for txt_file in sorted(raw_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": txt_file.stem, "text": text})
    print(f"Loaded {len(docs)} documents from {raw_dir}")
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 60]


def build_chunks(docs: list[dict]) -> tuple[list[str], list[dict]]:
    texts, metadata = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            texts.append(chunk)
            metadata.append({"source": doc["source"], "chunk_index": i})
    print(f"Created {len(texts)} chunks total")
    return texts, metadata


def build_index(texts: list[str], metadata: list[dict], index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    # Normalize so inner-product == cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadata": metadata}, f)

    print(f"Saved FAISS index — {index.ntotal} vectors, dim={embeddings.shape[1]}")


def run():
    docs = load_raw_files(RAW_DIR)
    if not docs:
        print("No raw files found. Run scraper.py first.")
        return
    texts, metadata = build_chunks(docs)
    build_index(texts, metadata, INDEX_DIR)
    print("Embedding complete.")


if __name__ == "__main__":
    run()
