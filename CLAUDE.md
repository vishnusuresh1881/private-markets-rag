# Private markets RAG assistant: context for Claude Code

## What this project is
A learning project I'm building to prepare for AI engineering and forward-deployed engineer
interviews. It's a RAG assistant that answers questions from private-markets documents (IPEV
Valuation Guidelines 2025, ILPA Principles 3.0, ILPA Reporting Template Definitions, ILPA
Template FAQs v1.1, 3i Group Annual Report FY2026), quotes its sources, and says "not found"
when the documents don't answer. An evaluation harness scores it so every change is measured.

My background: business application developer (Appian, Power Platform, Python, REST APIs),
moving into AI engineering. Interview feedback said I need more depth on engineering concepts,
production architecture and scaling, so understanding and explaining the code matters more
than speed.

## How I want you to help
- Act as a tutor. Before each command, give one sentence on WHY we're running it.
- Explain code in plain language, then the technical term.
- Don't write the hybrid search (RRF) for me. I'm writing it myself. Review my code, point out
  mistakes and hint, but let me fix them.
- Ask me questions to check I understand, as an interviewer would.
- Change one setting at a time in experiments, and check the config line in each result.
- Don't commit or push to GitHub without asking me first.

## Security rules
- Never read, print, log or write my ANTHROPIC_API_KEY. It lives only in an environment variable.
- Never put keys or tokens in code, config, commits or URLs.
- PDFs in docs/ are not committed (see .gitignore). Don't add them to git.

## Structure
- rag/config.py: all settings (chunk size, overlap, TOP_K, RETRIEVAL_MODE, models)
- rag/ingest.py: read PDFs, chunk per page with overlap, embed (all-MiniLM-L6-v2), save to index/
- rag/retrieve.py: score chunks by embeddings (cosine) or BM25, return top-k. Hybrid is my TODO.
- rag/answer.py: prompt Claude with retrieved chunks, validate the reply with Pydantic,
  check quotes are verbatim in code
- eval/run_eval.py: runs eval/test_set.json (28 cases: 18 answerable, 5 unanswerable,
  5 multi-passage); scores retrieval hit rate, correctness (LLM-as-judge), refusals, citations;
  saves results with the config used
- eval/compare.py: prints all saved runs side by side
- api/main.py: FastAPI service, POST /ask (X-API-Key) and GET /health
- notes.md: my findings so far (read this for the detail)

## Running it (Mac)
    source .venv/bin/activate
    export ANTHROPIC_API_KEY=...        # I type this myself
    python -m rag.ingest                # only after changing docs or chunk settings
    python -m rag.retrieve "question"
    python -m rag.answer "question"
    python -m eval.run_eval --no-llm --label NAME   # retrieval only, free
    python -m eval.run_eval --label NAME            # full run, uses the API
    python -m eval.compare
    # API (own terminal; it stays busy while running, Ctrl+C to stop)
    export RAG_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    uvicorn api.main:app --reload       # then open http://127.0.0.1:8000/docs
    # "Address already in use": an old uvicorn is still running; stop it first

## Results so far (retrieval hit rate, 23 answerable + multi-passage cases)
Day 3 metric fix: expected_phrases can now list alternative phrasings of the same fact (the
old exact-phrase check counted correct retrievals as misses). "Old" = old metric, kept for history.

| Run | Change | Old | Corrected | Still missed (corrected) |
|---|---|---|---|---|
| baseline | 350-word chunks, embeddings, top 4 | 57% | 70% | q10, q12, q17, q19, q24, q25, q27 (answers 70% correct, refusals 100%, citations 100%) |
| chunk180 | 180-word chunks | 61% | 74% | q12, q17, q19, q24, q25, q27 |
| chunk180-k8 | top 8 | 78% | 91% | q17, q27 |
| chunk180-bm25 | keyword search, top 4 | 65% | 78% | q02, q19, q25, q27, q28 |
| chunk180-hybrid | RRF of embeddings + BM25, top 4 | 74% | **96%** | q27 only |

Hybrid full run: answers 96% correct (100% partial or better), refusals 100%, citations 100% (70/70).
Corrected numbers come from the v2- result files in results/ (old-metric runs in results/old_metric/).
Current config: CHUNK_WORDS 180, OVERLAP 30, TOP_K 4, RETRIEVAL_MODE "hybrid".

## Next steps (in order)
1. Hybrid search: DONE (Day 3). 96% retrieval, 96% answers correct (full run done).
   Result files re-run with the corrected metric (v2- labels).
2. Optional experiment: the only remaining miss is q27, a two-part question. Query
   decomposition (one search per part) fits it better than a cross-encoder reranker, which
   was planned for q01/q06/q18 before those turned out to be metric false negatives.
3. README: results table, three failure analyses, known limits, and "what I'd change for
   production" (pgvector, reranking, monitoring, cost, access control, secrets management).
   Then make the GitHub repo public and add the link to my CV.
4. API: DONE (Day 4), api/main.py, tested end to end via /docs.
   - POST /ask (answer, citations, passages with scores, latency_ms) and GET /health
   - Pydantic request/response models (reuse Citation from rag/answer.py)
   - index, embedding model and BM25 loaded once at startup (lifespan + warm-up query)
   - X-API-Key vs RAG_API_KEY: fail closed, secrets.compare_digest (timing-safe)
   - 503 if Claude is unavailable or ANTHROPIC_API_KEY is missing (the library raises a
     plain TypeError for a missing key, so it is checked explicitly)
   - one log line per request (latency, found, citations); plain `def` endpoint so blocking
     calls run in FastAPI's thread pool
5. UI: a Streamlit page that CALLS THE API (not the Python code directly): question box,
   answer, quoted citations, retrieved passages with scores. Separate front end and back end.
6. Optional: package with Docker so it runs the same anywhere.
7. Later: turn it into an LPA review agent (checklist-driven, structured Pydantic report).

For each new step: explain the design choice and how I'd describe it in an interview.
