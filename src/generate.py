"""Prompt assembly and answer generation (Anthropic or OpenAI, via src.llm)."""

import logging
from dataclasses import dataclass

from src import llm
from src.config import GENERATION_MODEL, GENERATION_MODELS, LLM_PROVIDER, Provider
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
2. Give a complete answer, not just the headline number. Lead with the direct \
answer, then add the details the context states alongside it that change what it \
means for the employee: who it applies to, limits or caps, related totals or \
equivalents, timing and deadlines, and how it relates to other policies. Keep it \
concise and leave out details that don't bear on the question.
3. If the answer requires combining information from multiple excerpts, do so and \
cite all relevant sources.
4. If the answer is not found in the provided context, respond with exactly: \
"{REFUSAL_MESSAGE}"
5. Never speculate or infer beyond what the context explicitly states. Use the \
context's own wording for rules and conditions rather than paraphrasing them into \
something broader.\
"""

_CHUNK_PREVIEW_CHARS = 120

logger = logging.getLogger(__name__)


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


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    provider: Provider | None = None,
    api_key: str | None = None,
) -> GenerationResult:
    """Assemble the prompt, call the generation model, and return the answer with citations.

    provider defaults to LLM_PROVIDER; api_key defaults to that provider's .env key.
    """
    provider = provider or LLM_PROVIDER
    model = GENERATION_MODELS[provider]
    context_block = build_context_block(chunks)
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"

    logger.info("calling %s (%s) with %d context chunks", model, provider, len(chunks))
    completion = llm.complete(provider, model, SYSTEM_PROMPT, user_message, max_tokens=1024, api_key=api_key)
    answer = completion.text
    logger.info(
        "generation complete — input_tokens=%d output_tokens=%d",
        completion.input_tokens,
        completion.output_tokens,
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
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
