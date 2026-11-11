# Cited Q&A assistant with an evaluation harness

A small RAG assistant that answers questions from a set of public documents, cites its sources
with exact quotes, and says "not found" when the answer isn't there. An evaluation harness scores
it on a fixed test set so changes can be measured instead of guessed.

> Status: prototype. Not production. See "What I would change for production" (fill in on Day 3).

## Setup
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                            # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
```
Never put the key in code or commit it. `.env` and `docs/*.pdf` are git-ignored.

## Day 1 - build and try it
1. Put 4-6 public PDFs in `docs/` (check you can select text in each).
2. `python -m rag.ingest`  (reads, chunks, embeds, saves to `index/`)
3. `python -m rag.retrieve "your question"`  (look at what comes back - is it relevant?)
4. `python -m rag.answer "your question"`  (answer + citations + whether quotes are verbatim)
5. Ask 10 questions by hand. Note every failure in `notes.md`.

## Day 2 - evaluation
1. Fill in `eval/test_set.json`: ~25 cases (15 `answerable`, 5 `unanswerable`, 5 `multi_passage`).
   `expected_phrases` = short strings that must appear in the chunk(s) holding the answer.
2. `python -m eval.run_eval --no-llm --label retrieval-only`  (free, tests retrieval alone)
3. `python -m eval.run_eval --label baseline`  (full run; saves to `results/`)

Metrics: retrieval hit rate, answer correctness (LLM judge with a rubric - spot-check a few by hand),
refusal accuracy on unanswerable questions, and whether quoted citations are verbatim (checked in code).

## Day 3 - improve, then document
Change ONE setting in `rag/config.py` at a time (chunk size, TOP_K, RETRIEVAL_MODE, prompt), re-run
`python -m rag.ingest` if chunking changed, then `python -m eval.run_eval --label <what-you-changed>`.
`python -m eval.compare` prints the before/after table. Exercise: implement `hybrid` retrieval in
`rag/retrieve.py` (embeddings + BM25 combined with reciprocal rank fusion).

## Results
(Paste the comparison table here, then 3 failure analyses: what went wrong, why, what fixed it.)

## Known limits
(Be honest: test set size, LLM-judge noise, no OCR, local index, etc.)

## What I would change for production
(Architecture, scaling, reliability, operations - half a page.)
