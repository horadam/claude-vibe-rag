"""Query embedding and Chroma nearest-neighbour search."""

import logging
from dataclasses import dataclass

import chromadb
from openai import OpenAI

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    TOP_K,
)

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class RetrievedChunk:
    """A single result from the vector store."""

    doc_id: str
    text: str
    source: str
    chunk_index: int
    title: str
    distance: float


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Embed question and return the top_k nearest chunks from Chroma."""
    logger.info("embedding question (model=%s): %r", EMBEDDING_MODEL, question[:120])
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    query_embedding = response.data[0].embedding

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found — run ingest_cli.py first."
        ) from exc

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for doc_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                doc_id=doc_id,
                text=text,
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                title=meta["title"],
                distance=dist,
            )
        )

    logger.info(
        "retrieval returned %d chunks: %s",
        len(chunks),
        [f"{c.source}[{c.chunk_index}] d={c.distance:.4f}" for c in chunks],
    )
    return chunks
