"""Environment loading and project-wide constants."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# Providers
# Generation and judging can run on either provider; embeddings are always
# OpenAI, because the index is built from OpenAI vectors (Anthropic has no
# embeddings API).
Provider = Literal["anthropic", "openai"]
PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai")
API_KEY_ENV: dict[Provider, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _provider_env(name: str, default: str) -> Provider:
    """Read a provider name from the environment, rejecting unknown values."""
    value = os.environ.get(name, default).strip().lower()
    if value not in PROVIDERS:
        raise RuntimeError(f"{name}={value!r} is not supported; use one of {', '.join(PROVIDERS)}.")
    return value  # type: ignore[return-value]


LLM_PROVIDER: Provider = _provider_env("LLM_PROVIDER", "anthropic")
JUDGE_PROVIDER: Provider = _provider_env("JUDGE_PROVIDER", LLM_PROVIDER)

# Models
EMBEDDING_MODEL: str = "text-embedding-3-small"
# Each provider's judge is a larger model than its generator, so the eval
# isn't one model grading its own answers. Every name can be overridden in .env.
GENERATION_MODELS: dict[Provider, str] = {
    "anthropic": os.environ.get("ANTHROPIC_GENERATION_MODEL", "claude-sonnet-4-6"),
    "openai": os.environ.get("OPENAI_GENERATION_MODEL", "gpt-5-mini"),
}
JUDGE_MODELS: dict[Provider, str] = {
    "anthropic": os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-opus-5-5"),
    "openai": os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5"),
}
GENERATION_MODEL: str = GENERATION_MODELS[LLM_PROVIDER]
JUDGE_MODEL: str = JUDGE_MODELS[JUDGE_PROVIDER]

# Paths
CORPUS_DIR: Path = Path("corpus")
CHROMA_DIR: str = "./chroma_db"
EVAL_QUESTIONS_FILE: Path = Path("evals/questions.json")
EVAL_RUNS_DIR: Path = Path("evals/runs")
EVAL_RESULTS_FILE: Path = Path("evals/results.md")

# Chroma
COLLECTION_NAME: str = "hr_handbook"

# Chunking
CHUNK_TARGET_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 50
TIKTOKEN_ENCODING: str = "cl100k_base"
EMBED_BATCH_SIZE: int = 100

# Retrieval
TOP_K: int = 5


def require_env(name: str) -> str:
    """Return an environment variable, or raise a clear error if it's missing.

    Keys are read when a client is first needed, not at import, so offline
    code (tests, chunking, eval scoring) runs without them.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and add it.")
    return value


def has_env(name: str) -> bool:
    """Whether an environment variable is set to a non-empty value."""
    return bool(os.environ.get(name))
