"""Prompt assembly and Claude answer generation."""

import logging
from dataclasses import dataclass

import anthropic

from src.config import ANTHROPIC_API_KEY, GENERATION_MODEL
from src.retrieve import RetrievedChunk

SYSTEM_PROMPT = """\
You are the HR assistant for Northwind Robotics. Answer the employee's question \
using ONLY the context excerpts provided below. Do not use any outside knowledge.

Rules:
1. If the answer is present in the context, answer clearly and cite the source \
filename(s) in parentheses, e.g. (pto_policy.md).
2. If the answer requires combining information from multiple excerpts, do so and \
cite all relevant sources.
3. If the answer is not found in the provided context, respond with exactly: \
"I don't have that information in the handbook."
4. Never speculate or infer beyond what the context explicitly states.\
"""

_CHUNK_PREVIEW_CHARS = 120

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class Citation:
    source: str
    chunk_preview: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the CONTEXT section of the user message."""
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[source: {chunk.source}]\n{chunk.text}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    """Assemble the prompt, call Claude, and return the answer with citations."""
    context_block = _build_context_block(chunks)
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"

    logger.info("calling %s with %d context chunks", GENERATION_MODEL, len(chunks))
    message = _client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = message.content[0].text
    usage = message.usage
    logger.info(
        "generation complete — input_tokens=%d output_tokens=%d",
        usage.input_tokens,
        usage.output_tokens,
    )

    # build citations from chunks whose source appears in the answer
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        if chunk.source in answer and chunk.source not in seen:
            seen.add(chunk.source)
            preview = chunk.text[:_CHUNK_PREVIEW_CHARS].replace("\n", " ")
            citations.append(Citation(source=chunk.source, chunk_preview=preview))

    return GenerationResult(answer=answer, citations=citations)
