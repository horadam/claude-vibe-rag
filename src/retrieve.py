"""Query embedding and Chroma nearest-neighbour search."""

import logging
from dataclasses import dataclass
from functools import cache

import chromadb
from chromadb.errors import NotFoundError
from openai import OpenAI

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    require_env,
)

logger = logging.getLogger(__name__)



@cache
def _openai() -> OpenAI:
    """Return the shared OpenAI client, created on first use."""
    return OpenAI(api_key=require_env("OPENAI_API_KEY"))


@dataclass
class RetrievedChunk:
    """A single result from the vector store."""

    doc_id: str
    text: str
    source: str
    chunk_index: int
    title: str
    distance: float


def _collection() -> chromadb.Collection:
    """Open the handbook collection, or raise a clear error if it hasn't been built."""
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return chroma.get_collection(COLLECTION_NAME)
    except NotFoundError as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found — run ingest_cli.py first."
        ) from exc


def index_size() -> int:
    """Number of chunks in the index; 0 if ingestion hasn't run yet."""
    try:
        return _collection().count()
    except RuntimeError:
        # A missing collection is an expected state before the first ingest,
        # reported to the UI as an empty index rather than an error.
        return 0


def get_chunk(source: str, chunk_index: int) -> str | None:
    """Return the stored text of one chunk, or None if it doesn't exist."""
    doc_id = f"{source.removesuffix('.md')}__{chunk_index}"
    result = _collection().get(ids=[doc_id], include=["documents"])
    return result["documents"][0] if result["documents"] else None


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Embed question and return the top_k nearest chunks from Chroma."""
    # Open the collection first, so a missing index fails before a paid API call.
    collection = _collection()

    logger.info("embedding question (model=%s): %r", EMBEDDING_MODEL, question[:120])
    response = _openai().embeddings.create(model=EMBEDDING_MODEL, input=[question])
    query_embedding = response.data[0].embedding

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
