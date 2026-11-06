"""Prompt assembly and Claude answer generation."""

import logging
from dataclasses import dataclass
from functools import cache

import anthropic

from src.config import GENERATION_MODEL, require_env
from src.retrieve import RetrievedChunk

# The exact sentence the model must use when the handbook has no answer.
# The eval detects refusals by looking for it, so keep both in sync here.
REFUSAL_MESSAGE = "I don't have that information in the handbook."

SYSTEM_PROMPT = f"""\
You are the HR assistant for Northwind Robotics. Answer the employee's question \
using ONLY the context excerpts provided below. Do not use any outside knowledge.

Rules:
1. If the answer is present in the context, answer clearly and cite the source \
filename(s) in parentheses, e.g. (pto_policy.md).
2. If the answer requires combining information from multiple excerpts, do so and \
cite all relevant sources.
3. If the answer is not found in the provided context, respond with exactly: \
"{REFUSAL_MESSAGE}"
4. Never speculate or infer beyond what the context explicitly states.\
"""

_CHUNK_PREVIEW_CHARS = 120

logger = logging.getLogger(__name__)


@cache
def _client() -> anthropic.Anthropic:
    """Return the shared Anthropic client, created on first use."""
    return anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))


@dataclass
class Citation:
    source: str
    chunk_preview: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    model: str = GENERATION_MODEL
    input_tokens: int = 0
    output_tokens: int = 0


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the CONTEXT section of the user message."""
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[source: {chunk.source}]\n{chunk.text}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    """Assemble the prompt, call Claude, and return the answer with citations."""
    context_block = build_context_block(chunks)
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"

    logger.info("calling %s with %d context chunks", GENERATION_MODEL, len(chunks))
    message = _client().messages.create(
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

    return GenerationResult(
        answer=answer,
        citations=citations,
        model=message.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
