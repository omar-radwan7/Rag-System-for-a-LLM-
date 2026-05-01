#!/usr/bin/env python3
"""
rag_pipeline.py — Full RAG pipeline: embed query → retrieve from FAISS → prompt LLM.
"""

import json
import os
import re
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAG_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
FAISS_INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
METADATA_PATH = os.path.join(RAG_DIR, "chunk_metadata.json")

# Lazy-loaded globals
_embed_model = None
_faiss_index = None
_metadata = None

# System prompt to ensure the output matches the user's language
SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Reply in the SAME language as the user's question (Arabic → Arabic, English → English)."""


def clean_arabic_text(text: str) -> str:
    """Remove non-Arabic characters from full (non-streaming) model output."""
    cleaned = re.sub(
        r'[\u2E80-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]',
        '', text
    )
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()  # OK to strip for full text


def clean_chunk(chunk: str) -> str:
    """Remove non-Arabic characters from a streaming chunk WITHOUT stripping whitespace.
    
    IMPORTANT: Do NOT call .strip() here — each chunk may end with a space
    that separates it from the next word. Stripping removes that space and
    fuses words together (e.g. 'وصل ' + 'البطل' → 'وصلالبطل').
    """
    cleaned = re.sub(
        r'[\u2E80-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]',
        '', chunk
    )
    # Only collapse consecutive spaces — preserve single spaces
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned  # NO strip()


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        cfg = _load_config()
        model_name = cfg.get("embedding_model", "intfloat/multilingual-e5-small")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    return _faiss_index


def _get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    return _metadata


def retrieve(query: str, top_k: int = 5, dynamic_index: dict = None) -> tuple:
    """
    Embed query and retrieve top_k most similar chunks.
    If dynamic_index is provided (keys: 'index', 'metadata'), it is used
    instead of the disk-based index (for uploaded PDFs).
    Returns (chunks_list, retrieval_time_seconds).
    """
    model = _get_embed_model()

    if dynamic_index:
        index = dynamic_index["index"]
        metadata = dynamic_index["metadata"]
    else:
        index = _get_faiss_index()
        metadata = _get_metadata()

    start = time.time()
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, indices = index.search(q_emb, top_k)
    retrieval_time = round(time.time() - start, 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            chunk = metadata[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

    return results, retrieval_time


# Arabic character name & concept aliases found in the book
_NAME_ALIASES = {
    "المؤلف": ["عمرو عبد الحميد"],
    "الكاتب": ["عمرو عبد الحميد"],
    "البطل": ["خالد", "خالد حسني"],
    "الشخصية الرئيسية": ["خالد", "خالد حسني"],
    "بطل القصة": ["خالد", "خالد حسني"],
    "الجد": ["عبدو", "جد خالد"],
    "حبيبته": ["منى"],
    # How Khaled got to Ard Zikola — via the underground tunnel
    "أرض زيكولا": ["السرداب", "خالد السرداب"],
    "وصول": ["السرداب", "نزل السرداب"],
    "كيف وصل": ["السرداب", "نزل السرداب خالد"],
}

def _expand_query(query: str) -> list:
    """Returns a list of search queries: original + alias expansions."""
    queries = [query]
    for name, aliases in _NAME_ALIASES.items():
        if name in query:
            for alias in aliases:
                queries.append(query.replace(name, alias))
    return queries


def retrieve_multi(query: str, top_k: int = 5, dynamic_index: dict = None) -> tuple:
    """
    Search using multiple query variants and merge + deduplicate results.
    If dynamic_index is provided, skips alias expansion (book-agnostic mode).
    Returns (chunks_list, retrieval_time).
    """
    # Only use aliases for the hardcoded book; skip for uploaded PDFs
    queries = [query] if dynamic_index else _expand_query(query)
    seen_ids = set()
    all_chunks = []
    start = time.time()

    for q in queries:
        chunks, _ = retrieve(q, top_k=top_k, dynamic_index=dynamic_index)
        for c in chunks:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                all_chunks.append(c)

    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    retrieval_time = round(time.time() - start, 3)
    return all_chunks[:top_k], retrieval_time


def build_rag_prompt(question: str, chunks: list, history: list = None,
                    book_name: str = None) -> str:
    """
    Build the RAG prompt with 3-way behavior:
    1. Book question + in context  → answer from context
    2. Book question + not in context → answer from general knowledge
    3. Completely unrelated question → politely refuse
    """
    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    hist_str = ""
    if history:
        hist_str = "Previous conversation:\n"
        for msg in history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            hist_str += f"{role}: {msg['content']}\n"
        hist_str += "\n"

    book_intro = f'"{book_name}"' if book_name else "the uploaded book"

    prompt = f"""You are a dedicated book assistant specializing in {book_intro}.

YOUR RULES:
1. You only discuss topics related to this book — its plot, characters, themes, author, genre, writing style, or anything a reader might reasonably ask about it.
2. If the user asks about something unrelated to the book (e.g., cooking, coding, sports, general trivia), politely decline and redirect them: say something like "I'm here to help you with {book_intro}. Is there something about the book you'd like to know?"
3. To answer book-related questions, first check the context passages below. If the answer is there, use it.
4. If the context does NOT contain the answer but the question is still about the book, use your general knowledge about books, storytelling, and the topic to give a helpful answer — do not say "not found".
5. Always reply in the SAME LANGUAGE as the user's question.
6. Use the conversation history to resolve pronouns and follow-up references.

{hist_str}Context passages from the book (use when relevant):
{context}

User question:
{question}

Answer:"""
    return prompt


def _build_followup_query(question: str, history: list) -> str:
    """
    Build a richer search query for follow-up questions by including context
    from both the previous user question AND the assistant's answer.
    """
    if not history or len(history) < 2:
        return question

    # Only enhance short questions (likely follow-ups with pronouns)
    if len(question.split()) >= 15:
        return question

    # Gather context from recent history
    last_user_q = ""
    last_assistant_a = ""

    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant_a:
            # Take first 100 chars of assistant answer as context
            last_assistant_a = msg["content"][:100]
        elif msg["role"] == "user" and not last_user_q:
            last_user_q = msg["content"]
        if last_user_q and last_assistant_a:
            break

    # Combine: previous question + key part of answer + current question
    parts = []
    if last_user_q:
        parts.append(last_user_q)
    if last_assistant_a:
        parts.append(last_assistant_a)
    parts.append(question)

    return " ".join(parts)


def ask(question: str, model: str = None, top_k: int = 2, history: list = None,
        dynamic_index: dict = None, book_name: str = None) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.
    Pass dynamic_index + book_name for uploaded PDFs.
    """
    from rag.ollama_client import generate

    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    search_query = _build_followup_query(question, history)
    effective_top_k = top_k + 1 if history and len(question.split()) < 10 else top_k

    chunks, retrieval_time = retrieve_multi(search_query, top_k=effective_top_k,
                                            dynamic_index=dynamic_index)
    chunk_ids = [c["chunk_id"] for c in chunks]

    rag_prompt = build_rag_prompt(question, chunks, history, book_name=book_name)
    start = time.time()
    result = generate(model, rag_prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=512)
    generation_time = round(time.time() - start, 3)

    return {
        "answer": result["text"],
        "sources": chunk_ids,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "model": model,
        "chunks": chunks,
    }

def ask_stream(question: str, model: str = None, top_k: int = 2, history: list = None,
               dynamic_index: dict = None, book_name: str = None):
    from rag.ollama_client import generate_stream

    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    search_query = _build_followup_query(question, history)
    effective_top_k = top_k + 1 if history and len(question.split()) < 10 else top_k

    chunks, retrieval_time = retrieve_multi(search_query, top_k=effective_top_k,
                                            dynamic_index=dynamic_index)
    chunk_ids = [c["chunk_id"] for c in chunks]

    rag_prompt = build_rag_prompt(question, chunks, history, book_name=book_name)
    start = time.time()
    raw_stream = generate_stream(model, rag_prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=512)

    return raw_stream, chunk_ids, retrieval_time, start


def ask_llm_only(question: str, model: str = None) -> dict:
    from rag.ollama_client import generate
    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    prompt = f"""أجب بشكل مختصر جدا وبنفس لغة السؤال (لا تتجاوز 100 كلمة):\n\nالسؤال:\n{question}\n\nالإجابة:"""
    start = time.time()
    result = generate(model, prompt, temperature=0.3, max_tokens=150)
    generation_time = round(time.time() - start, 3)

    return {
        "answer": result["text"],
        "sources": [],
        "retrieval_time": 0,
        "generation_time": generation_time,
        "model": model,
    }

def ask_llm_only_stream(question: str, model: str = None):
    from rag.ollama_client import generate_stream
    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    prompt = f"""أجب بشكل مختصر جدا وبنفس لغة السؤال (لا تتجاوز 100 كلمة):\n\nالسؤال:\n{question}\n\nالإجابة:"""
    start = time.time()
    stream = generate_stream(model, prompt, temperature=0.3, max_tokens=150)
    
    return stream, [], 0, start

if __name__ == "__main__":
    q = "ما هي الفكرة الرئيسية في الكتاب؟"
    print("RAG answer:")
    result = ask(q)
    print(result["answer"][:300])
    print(f"Sources: {result['sources']}")
    print(f"Retrieval: {result['retrieval_time']}s, Generation: {result['generation_time']}s")
