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

## Results so far (retrieval hit rate, 23 answerable + multi-passage cases)
| Run | Change | Retrieval |
|---|---|---|
| baseline | 350-word chunks, embeddings, top 4 | 57% (answers 70% correct, refusals 100%, citations 100%) |
| chunk180 | 180-word chunks | 61% (only q10 flipped: truncation theory only partly right) |
| chunk180-k8 | top 8 | 78% (confirms crowding: right passage ranked 5th-8th) |
| chunk180-bm25 | keyword search, top 4 | 65% (wins on specific terms q12/q17/q24, loses q01/q02) |
| chunk180-hybrid | RRF of embeddings + BM25, top 4 | 74% (fixes q25 and q02; loses q01: BM25 drags the answer chunk from rank 4 to 8) |

Missed by hybrid: q01, q06, q18, q19, q27, q28 (q25 was the only "missed by both" case fusion fixed).
Current config: CHUNK_WORDS 180, OVERLAP 30, TOP_K 4, RETRIEVAL_MODE "hybrid".

## Next steps (in order)
1. Hybrid search: DONE (Day 3), `_rrf` and the hybrid branch in rag/retrieve.py, 74% at top 4.
   Still to do: run the full evaluation (`--label chunk180-hybrid-full`).
2. Optional experiment: cross-encoder reranker (retrieve 20, rerank to 4). Likely fix for
   crowding misses like q06 and q18.
3. README: results table, three failure analyses, known limits, and "what I'd change for
   production" (pgvector, reranking, monitoring, cost, access control, secrets management).
   Then make the GitHub repo public and add the link to my CV.
4. API: wrap the pipeline in FastAPI.
   - POST /ask (question in; answer, citations, retrieved passages out) and GET /health
   - Pydantic request/response models (reuse AnswerResult)
   - load the index once at startup, not per request
   - error handling, logging of each question and its latency, an API key on the endpoint
     (read from an environment variable)
   - try it via the auto-generated docs at /docs
5. UI: a Streamlit page that CALLS THE API (not the Python code directly): question box,
   answer, quoted citations, retrieved passages with scores. Separate front end and back end.
6. Optional: package with Docker so it runs the same anywhere.
7. Later: turn it into an LPA review agent (checklist-driven, structured Pydantic report).

For each new step: explain the design choice and how I'd describe it in an interview.
