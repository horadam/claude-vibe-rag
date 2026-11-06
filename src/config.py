"""Environment loading and project-wide constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Models
EMBEDDING_MODEL: str = "text-embedding-3-small"
GENERATION_MODEL: str = "claude-sonnet-4-6"
# The judge is a different, larger model than the generator, so the eval
# isn't one model grading its own answers. Override with JUDGE_MODEL in .env.
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")

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
