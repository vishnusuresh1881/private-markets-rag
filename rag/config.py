"""Settings in one place. Day 3: change ONE of these at a time, re-run the eval, log the result."""
import os

CHUNK_WORDS = 180        # roughly 240 tokens (embedding model limit is 256)
CHUNK_OVERLAP_WORDS = 30
TOP_K = 4
RETRIEVAL_MODE = "hybrid"  # "embed" or "bm25" (Day 3 exercise: add "hybrid")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Model name comes from the environment so you can change it without editing code.
LLM_MODEL = os.environ.get("RAG_MODEL", "claude-sonnet-5-5")
DOCS_DIR = "docs"
INDEX_DIR = "index"
