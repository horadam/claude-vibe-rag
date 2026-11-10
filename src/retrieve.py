"""Query embedding and nearest-neighbour search over the exported JSON index.

Chroma builds the index at ingest time; queries read the JSON export instead,
so the server needs neither Chroma nor a writable disk. With a few dozen
chunks, an exact search is a short loop of dot products.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src import llm
from src.config import (
    EMBEDDING_MODEL,
    INDEX_FILE,
    TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single result from the vector store."""

    doc_id: str
    text: str
    source: str
    chunk_index: int
    title: str
    distance: float


@dataclass
class IndexedChunk:
    """One chunk of the exported index, with its embedding."""

    doc_id: str
    text: str
    source: str
    chunk_index: int
    title: str
    embedding: list[float]


# The parsed index, keyed by the file's modification time, so a re-ingest is
# picked up without restarting the server.
_cache: tuple[float, list[IndexedChunk]] | None = None


def load_index(path: Path = INDEX_FILE) -> list[IndexedChunk]:
    """Return every indexed chunk, or raise a clear error if the index hasn't been built."""
    global _cache
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError as exc:
        raise RuntimeError(f"Index file {path} not found — run ingest_cli.py first.") from exc
    if _cache is not None and _cache[0] == mtime:
        return _cache[1]

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [IndexedChunk(**item) for item in data["chunks"]]
    _cache = (mtime, chunks)
    return chunks


def squared_l2(a: list[float], b: list[float]) -> float:
    """Squared Euclidean distance, the measure Chroma's default index reports.

    OpenAI embeddings have length 1, so this equals 2 × (1 − cosine similarity)
    and ranks chunks exactly as cosine similarity would.
    """
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def search(query_embedding: list[float], chunks: list[IndexedChunk], top_k: int) -> list[RetrievedChunk]:
    """Return the top_k chunks nearest to query_embedding, closest first."""
    scored = sorted(
        ((squared_l2(query_embedding, c.embedding), c) for c in chunks),
        key=lambda pair: pair[0],
    )
    return [
        RetrievedChunk(
            doc_id=c.doc_id,
            text=c.text,
            source=c.source,
            chunk_index=c.chunk_index,
            title=c.title,
            distance=dist,
        )
        for dist, c in scored[:top_k]
    ]


def index_size() -> int:
    """Number of chunks in the index; 0 if ingestion hasn't run yet."""
    try:
        return len(load_index())
    except RuntimeError:
        # A missing index is an expected state before the first ingest,
        # reported to the UI as an empty index rather than an error.
        return 0


def get_chunk(source: str, chunk_index: int) -> str | None:
    """Return the stored text of one chunk, or None if it doesn't exist."""
    for c in load_index():
        if c.source == source and c.chunk_index == chunk_index:
            return c.text
    return None


def retrieve(question: str, top_k: int = TOP_K, api_key: str | None = None) -> list[RetrievedChunk]:
    """Embed question and return the top_k nearest chunks from the index.

    api_key is an OpenAI key for the query embedding; None uses the .env key.
    """
    # Load the index first, so a missing index fails before a paid API call.
    chunks = load_index()

    logger.info("embedding question (model=%s): %r", EMBEDDING_MODEL, question[:120])
    response = llm.openai_client(api_key).embeddings.create(model=EMBEDDING_MODEL, input=[question])
    query_embedding = response.data[0].embedding

    results = search(query_embedding, chunks, top_k)
    logger.info(
        "retrieval returned %d chunks: %s",
        len(results),
        [f"{c.source}[{c.chunk_index}] d={c.distance:.4f}" for c in results],
    )
    return results
