"""Step 4: a FastAPI service around the RAG pipeline.

Run:   export RAG_API_KEY=some-long-random-string     # you choose it; never commit it
       export ANTHROPIC_API_KEY=...                    # needed for /ask
       uvicorn api.main:app --reload
Try:   http://127.0.0.1:8000/docs
"""
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag.answer import Citation, answer
from rag.retrieve import _load, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag.api")


# --- Request and response models: the API's contract -----------------------

class AskRequest(BaseModel):
    """What a client must send."""
    question: str = Field(min_length=3, max_length=500,
                          examples=["What is the definition of fair value?"])


class Passage(BaseModel):
    """One retrieved chunk, so the caller can see WHY the model answered as it did."""
    id: int        # chunk id from ingest; citations point at this
    doc: str       # PDF file name
    page: int      # page number in that PDF
    score: float   # retrieval score (an RRF score in hybrid mode, about 0.03)
    text: str      # the chunk text that was sent to the model


class AskResponse(BaseModel):
    """What the API sends back."""
    # From answer() in rag/answer.py
    found: bool                        # False means "the documents don't answer this"
    answer: str
    citations: list[Citation] = []     # reuses the model answer.py already validates with
    citations_valid: int               # quotes found verbatim in the cited chunk
    citations_total: int
    # Added by the API
    passages: list[Passage]
    latency_ms: int
    # Extra keys in answer()'s dict ("retrieved", "parse_error") are ignored by Pydantic.


# --- Startup: load everything once, not on the first request ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before `yield` runs once at startup; code after it runs at shutdown.
    if not os.environ.get("RAG_API_KEY"):
        log.warning("RAG_API_KEY is not set: every /ask request will be refused (401)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY is not set: /ask will return 503")
    # A pasted key with a trailing newline makes every Claude call fail with a vague
    # "Connection error" (newlines are illegal in HTTP headers). Check names only, never values.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "RAG_API_KEY"):
        value = os.environ.get(name)
        if value and value != value.strip():
            log.warning("%s has leading/trailing whitespace or a newline: re-set it", name)
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        log.warning("ANTHROPIC_AUTH_TOKEN is set: the SDK will also send it as a Bearer token")
    retrieve("warm up")              # loads chunks, embeddings, embedding model, BM25 into the cache
    chunks, _ = _load()
    log.info("Index loaded: %d chunks", len(chunks))
    yield
    log.info("Shutting down")


app = FastAPI(title="Private markets RAG API", version="0.1.0", lifespan=lifespan)


# --- GET /health: "is this service alive?" ----------------------------------

@app.get("/health")
def health():
    # No API key, no Claude call: load balancers call this often and it must be cheap.
    chunks, _ = _load()
    return {"status": "ok", "chunks": len(chunks)}


# --- API key check (a FastAPI dependency) -----------------------------------

def require_api_key(x_api_key: str | None = Header(default=None)):
    """FastAPI reads the X-API-Key header into x_api_key. Raising here stops the request."""
    expected = os.environ.get("RAG_API_KEY")
    # Fail closed: no configured key, no header, or a wrong key all mean "no".
    # compare_digest takes the same time however many characters match, so an attacker
    # can't guess the key one character at a time by measuring response times.
    if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- POST /ask ----------------------------------------------------------------
# Plain `def`, NOT `async def`: answer() blocks (CPU embedding + a synchronous Claude
# call). FastAPI runs plain `def` endpoints in a thread pool; a blocking call inside
# `async def` would freeze the server for every other request.

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest):
    start = time.perf_counter()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # The anthropic library raises a plain TypeError for a missing key, so check it
        # here rather than catching every TypeError (which would hide real bugs).
        log.error("ask refused: ANTHROPIC_API_KEY is not set")
        raise HTTPException(status_code=503, detail="Language model unavailable, try again")
    try:
        result, chunks = answer(req.question)
    except anthropic.AnthropicError:
        # Covers Claude being down, timeouts and rate limits.
        # Log the full error for us; give the caller a clean message with no internals.
        log.exception("Claude call failed")
        raise HTTPException(status_code=503, detail="Language model unavailable, try again")

    passages = [Passage(**c) for c in chunks]
    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info("ask latency_ms=%d found=%s citations=%d/%d question=%r",
             latency_ms, result["found"], result["citations_valid"],
             result["citations_total"], req.question)
    return AskResponse(**result, passages=passages, latency_ms=latency_ms)
