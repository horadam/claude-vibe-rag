"""Chunking, embedding, and Chroma upsert for the HR handbook corpus."""

import logging
from pathlib import Path

import tiktoken
from openai import OpenAI

import chromadb

from src.config import (
    ANTHROPIC_API_KEY,
    CHROMA_DIR,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    COLLECTION_NAME,
    CORPUS_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    TIKTOKEN_ENCODING,
)

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=OPENAI_API_KEY)
_encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _token_count(text: str) -> int:
    """Return the number of tokens in text."""
    return len(_encoding.encode(text))


def _extract_title(text: str) -> str:
    """Return the first H1 heading, or the filename stem if none found."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def chunk_document(text: str) -> list[str]:
    """Split document text into overlapping chunks of ~CHUNK_TARGET_TOKENS tokens.

    Strategy: accumulate paragraphs greedily; when the next paragraph would
    exceed the target, finalise the chunk and carry the last paragraph forward
    as best-effort overlap (skipped if that paragraph is itself over 2x the
    overlap budget).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _token_count(para)

        if current and current_tokens + para_tokens > CHUNK_TARGET_TOKENS:
            chunks.append("\n\n".join(current))
            # carry the last paragraph as overlap if it fits the budget
            last_para = current[-1]
            last_tokens = _token_count(last_para)
            if last_tokens <= CHUNK_OVERLAP_TOKENS * 2:
                current = [last_para, para]
                current_tokens = last_tokens + para_tokens
            else:
                current = [para]
                current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured OpenAI model."""
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest_corpus(corpus_dir: Path = CORPUS_DIR) -> None:
    """Read, chunk, embed, and upsert all markdown files in corpus_dir."""
    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        logger.error("No markdown files found in %s", corpus_dir)
        raise SystemExit(1)

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    # idempotent: wipe and recreate the collection
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(COLLECTION_NAME)

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    total_chunks = 0
    total_tokens = 0

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        title = _extract_title(text)
        chunks = chunk_document(text)
        file_tokens = sum(_token_count(c) for c in chunks)

        print(f"  {md_file.name}: {len(chunks)} chunks, {file_tokens} tokens")
        total_chunks += len(chunks)
        total_tokens += file_tokens

        for idx, chunk in enumerate(chunks):
            doc_id = f"{md_file.stem}__{idx}"
            all_ids.append(doc_id)
            all_documents.append(chunk)
            all_metadatas.append(
                {
                    "source": md_file.name,
                    "chunk_index": idx,
                    "title": title,
                }
            )

    # embed in batches
    print(f"\nEmbedding {total_chunks} chunks in batches of {EMBED_BATCH_SIZE}...")
    embeddings: list[list[float]] = []
    for start in range(0, len(all_documents), EMBED_BATCH_SIZE):
        batch = all_documents[start : start + EMBED_BATCH_SIZE]
        embeddings.extend(_embed_batch(batch))
        print(f"  embedded {min(start + EMBED_BATCH_SIZE, len(all_documents))}/{len(all_documents)}")

    # upsert everything
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    print(f"\nDone. {total_chunks} chunks, {total_tokens} tokens ingested into '{COLLECTION_NAME}'.")
