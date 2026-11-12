"""Day 1, step 2: given a question, return the top-k most relevant chunks."""
import json
import pathlib
import re
import numpy as np
from . import config

_cache = {}


def _load():
    if "chunks" not in _cache:
        idx = pathlib.Path(config.INDEX_DIR)
        _cache["chunks"] = json.loads((idx / "chunks.json").read_text(encoding="utf-8"))
        emb = idx / "embeddings.npy"
        _cache["emb"] = np.load(emb) if emb.exists() else None
    return _cache["chunks"], _cache["emb"]


def _tokenise(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query: str, chunks):
    from rank_bm25 import BM25Okapi
    if "bm25" not in _cache:
        _cache["bm25"] = BM25Okapi([_tokenise(c["text"]) for c in chunks])
    return _cache["bm25"].get_scores(_tokenise(query))


def _embed_scores(query: str, emb):
    from sentence_transformers import SentenceTransformer
    if "model" not in _cache:
        _cache["model"] = SentenceTransformer(config.EMBED_MODEL)
    q = _cache["model"].encode([query], normalize_embeddings=True)[0]
    return emb @ q  # cosine similarity because vectors are normalised

def _rrf(score_lists, rrf_k: int = 60):
    """Reciprocal rank fusion: combine several score arrays (one score per chunk,
    same chunk order) into one fused array. Uses ranks, not raw scores, so methods
    with different scales (cosine 0-1, BM25 unbounded) can be combined fairly."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        order = np.argsort(scores)[::-1]
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (rrf_k + ranks)
    return fused


def retrieve(query: str, k: int = None, mode: str = None):
    """Return a list of chunk dicts (with a 'score' added), best first."""
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    chunks, emb = _load()
    if mode == "embed":
        if emb is None:
            raise SystemExit("No embeddings found. Run `python -m rag.ingest` first "
                             "(or set RETRIEVAL_MODE='bm25').")
        scores = _embed_scores(query, emb)
    elif mode == "bm25":
        scores = _bm25_scores(query, chunks)
    elif mode == "hybrid":
        if emb is None:
            raise SystemExit("No embeddings found. Run `python -m rag.ingest` first.")
        scores = _rrf([_embed_scores(query, emb), _bm25_scores(query, chunks)])
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    top = np.argsort(scores)[::-1][:k]
    return [dict(chunks[i], score=float(scores[i])) for i in top]


if __name__ == "__main__":
    import sys
    for c in retrieve(" ".join(sys.argv[1:]) or "test"):
        print(f"[{c['doc']} p{c['page']} score={c['score']:.3f}] {c['text'][:200]}...\n")
