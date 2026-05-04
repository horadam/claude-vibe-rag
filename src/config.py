"""Environment loading and project-wide constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API keys
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

# Models
EMBEDDING_MODEL: str = "text-embedding-3-small"
GENERATION_MODEL: str = "claude-sonnet-4-6"

# Paths
CORPUS_DIR: Path = Path("corpus")
CHROMA_DIR: str = "./chroma_db"

# Chroma
COLLECTION_NAME: str = "hr_handbook"

# Chunking
CHUNK_TARGET_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 50
TIKTOKEN_ENCODING: str = "cl100k_base"
EMBED_BATCH_SIZE: int = 100

# Retrieval
TOP_K: int = 5
