#!/usr/bin/env python3
"""
pdf_indexer.py — Extract text from a PDF, chunk it, and build an in-memory
FAISS index so any book can be loaded at runtime without touching disk indexes.
"""

import io
import re
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

CHUNK_SIZE = 300    # target words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str) -> list:
    """
    Split text into overlapping word-based chunks.
    Returns a list of dicts: {chunk_id, text, word_count}
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "word_count": len(chunk_words),
            "chapter": "",
        })
        chunk_id += 1
        # Advance by CHUNK_SIZE - CHUNK_OVERLAP for overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP
        if end == len(words):
            break

    return chunks


def build_memory_index(pdf_bytes: bytes) -> dict:
    """
    Full pipeline: PDF bytes → text → chunks → embeddings → in-memory FAISS index.
    Returns a dict with keys: 'index', 'metadata', 'book_name'.
    The returned object can be passed directly to the RAG pipeline.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = _load_config()
    embed_model_name = cfg.get("embedding_model", "intfloat/multilingual-e5-small")

    # Extract text
    full_text = extract_text_from_pdf(pdf_bytes)
    if not full_text.strip():
        raise ValueError("Could not extract any text from the PDF. It may be a scanned image.")

    # Chunk
    chunks = chunk_text(full_text)
    if not chunks:
        raise ValueError("PDF text was extracted but produced no chunks.")

    # Embed
    # Reuse the already-loaded model from rag_pipeline globals if available
    from rag.rag_pipeline import _get_embed_model
    model = _get_embed_model()

    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (in-memory only)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "chapter": c.get("chapter", ""),
            "text": c["text"],
            "word_count": c["word_count"],
        }
        for c in chunks
    ]

    return {
        "index": index,
        "metadata": metadata,
        "chunk_count": len(chunks),
        "text_preview": full_text[:200],
    }
