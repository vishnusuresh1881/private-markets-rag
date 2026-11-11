# Notes

## Day 1 (10 Nov 2026): build and first tests

Corpus: 2025 IPEV Valuation Guidelines, ILPA Principles 3.0, ILPA Reporting Template Definitions,
ILPA Template FAQs v1.1, 3i Group Annual Report FY2026. 760 chunks, 384-dimension embeddings.
(Left out: ILPA Reporting Template Suggested Guidance, mostly tables. Try adding it on Day 3.)

### 1. Query phrasing changed retrieval
- "What is fair value under the IPEV guidelines?" retrieved the intro (p5), the 2025 changes
  appendix (p87), a 3i policy page and the disclaimer (p2). None held the definition.
- "What is the definition of fair value?" retrieved the definition (p75, p10).
- Cause: the document name in the question matched generic front matter.
- Ideas: query rewriting, keyword (BM25) or hybrid search, dropping front matter.

### 2. Why top-k is more than 1
- "What was 3i's total return for the year?" The right passage (p4, 22%) ranked 2nd behind a
  navigation page (p175). Scores were lower (~0.55 vs ~0.65).
- With top-k = 1 this would have failed. Trade-off: missing the answer vs adding noise and cost.

### 3. Text extraction quality
- "Reporting T emplate": a stray space from PDF extraction. Can hurt keyword search and the
  quote check.
- 3i report: one page with no extractable text (probably an image page).
- 3i KPI boxes were merged in extraction: "22% 84.5p Total return..." (84.5p is a different figure).

### 4. The quote check proves existence, not support
- The 3i answer's quote passed the verbatim check but contained the unrelated 84.5p from the
  merged KPI boxes. The answer itself was right.
- A deterministic check shows the text exists in the source, not that it supports the claim.
  Production systems add a second check (model or person).

### 5. Bug: thinking block crash
- The code read msg.content[0].text. On a harder question the first block was a thinking block,
  so it crashed (AttributeError: 'ThinkingBlock' object has no attribute 'text').
- Fix: join only the text blocks; raise max_tokens to 2000 (and the judge's to 1000).

### 6. Refusals
- "What is the carried interest rate in 3i's funds?" was meant to be unanswerable, but the
  report does discuss carry (c.10-12% of GIR on PE vintages). Claude answered with a correct
  caveat. Lesson: check with Ctrl+F before calling a question unanswerable.
- A yes/no "found" is too blunt; a "partial" outcome would be more honest.
- "What is the UK statutory maternity pay rate?" correctly returned found: false.
- Retrieval always returns k chunks, even when nothing is relevant. Idea: a similarity
  threshold to refuse before calling the model.

### 7. Chunk size vs the embedding model's limit
- all-MiniLM-L6-v2 reads at most 256 tokens (~190 words) and silently truncates the rest.
- With 350-word chunks, only about the first half of each chunk counts for retrieval.
- Day 3 experiment: CHUNK_WORDS = 180, CHUNK_OVERLAP_WORDS = 30, compared with the baseline.

## Day 2

## Day 3
