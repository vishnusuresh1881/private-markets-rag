# Private markets RAG assistant: context for Claude Code

## What this project is
A RAG assistant that answers questions from private-markets documents (IPEV Valuation
Guidelines 2025, ILPA Principles 3.0, ILPA Reporting Template Definitions, ILPA Template FAQs
v1.1, 3i Group Annual Report FY2026), quotes its sources, and says "not found" when the
documents don't answer. An evaluation harness scores it so every change is measured.
See README.md for results and design decisions, notes.md for the experiment log.

## Working conventions
- Change one setting at a time in experiments, and check the config line in each result
  (the label alone can't be trusted; it once hid a mislabelled run).
- Retrieval-only runs (`--no-llm`) first; full runs only when retrieval improves.
- Don't commit or push to GitHub without asking first.

## Security rules
- Never read, print, log or write ANTHROPIC_API_KEY or RAG_API_KEY. They live only in
  environment variables / the git-ignored .env file.
- Never put keys or tokens in code, config, commits or URLs.
- Library tracebacks can include request headers: check for `sk-ant` before sharing them.
- PDFs in docs/ are not committed (see .gitignore). Don't add them to git.

## Structure
- rag/config.py: all settings (chunk size, overlap, TOP_K, RETRIEVAL_MODE, models)
- rag/ingest.py: read PDFs, chunk per page with overlap, embed (all-MiniLM-L6-v2), save to index/
- rag/retrieve.py: score chunks by embeddings (cosine), BM25, or hybrid (reciprocal rank
  fusion, `_rrf`); return top-k
- rag/answer.py: prompt Claude with retrieved chunks, validate the reply with Pydantic,
  check quotes are verbatim in code
- eval/run_eval.py: runs eval/test_set.json (28 cases: 18 answerable, 5 unanswerable,
  5 multi-passage); scores retrieval hit rate, correctness (LLM-as-judge), refusals, citations;
  saves results with the config used. An expected phrase can be a list of alternatives.
- eval/compare.py: prints all saved runs side by side
- api/main.py: FastAPI service, POST /ask (X-API-Key) and GET /health
- ui/app.py: Streamlit front end; calls the API over HTTP, never imports rag/
- notes.md: experiment log, including wrong turns

## Running it (Mac)
    source .venv/bin/activate
    # Anthropic key: set without echo or trailing newline (paste, then Enter; run this line alone)
    read -s "ANTHROPIC_API_KEY?Anthropic key: "; export ANTHROPIC_API_KEY; echo
    python -m rag.ingest                # only after changing docs or chunk settings
    python -m rag.retrieve "question"
    python -m rag.answer "question"
    python -m eval.run_eval --no-llm --label NAME   # retrieval only, free
    python -m eval.run_eval --label NAME            # full run, uses the API
    python -m eval.compare
    # Once: RAG_API_KEY in .env (git-ignored) so API and UI get the identical key
    python -c "import secrets; print('export RAG_API_KEY=' + secrets.token_urlsafe(32))" > .env
    # Terminal 1, API (stays busy while running, Ctrl+C to stop):
    source .env && uvicorn api.main:app --reload    # http://127.0.0.1:8000/docs
    # Terminal 2, UI:
    source .env && streamlit run ui/app.py          # http://localhost:8501
    # "Address already in use": an old uvicorn is still running; stop it first
    # Programs read env vars at start: after changing a key, restart what uses it

## Results (retrieval hit rate, 23 answerable + multi-passage cases)
Metric fix (Day 3): expected_phrases can list alternative phrasings of the same fact (the
old exact-phrase check counted correct retrievals as misses). "Old" = old metric, for history.

| Run | Change | Old | Corrected | Still missed (corrected) |
|---|---|---|---|---|
| baseline | 350-word chunks, embeddings, top 4 | 57% | 70% | q10, q12, q17, q19, q24, q25, q27 (answers 70% correct, refusals 100%, citations 100%) |
| chunk180 | 180-word chunks | 61% | 74% | q12, q17, q19, q24, q25, q27 |
| chunk180-k8 | top 8 | 78% | 91% | q17, q27 |
| chunk180-bm25 | keyword search, top 4 | 65% | 78% | q02, q19, q25, q27, q28 |
| chunk180-hybrid | RRF of embeddings + BM25, top 4 | 74% | **96%** | q27 only |

Hybrid full run: answers 96% correct (100% partial or better), refusals 100%, citations 100% (70/70).
Result files: v2- runs in results/ (old-metric runs in results/old_metric/; results/ is git-ignored).
Current config: CHUNK_WORDS 180, OVERLAP 30, TOP_K 4, RETRIEVAL_MODE "hybrid".
