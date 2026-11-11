"""Day 1, step 3: ask the model to answer ONLY from the retrieved chunks, with citations.

Needs ANTHROPIC_API_KEY in your environment (never put the key in code or commit it).
Run:  python -m rag.answer "your question"
"""
import json
import re
from pydantic import BaseModel, ValidationError
from . import config
from .retrieve import retrieve

SYSTEM = """You answer questions using ONLY the numbered context passages provided.
Rules:
- If the passages do not contain the answer, set "found" to false and say you cannot find it. Do not guess.
- Every claim must be supported by a short quote copied EXACTLY (word for word) from a passage.
- Reply with JSON only, in this shape:
{"found": true/false, "answer": "...", "citations": [{"chunk_id": 12, "quote": "exact words from the passage"}]}"""

class Citation(BaseModel):
    chunk_id: int
    quote: str


class AnswerResult(BaseModel):
    found: bool
    answer: str
    citations: list[Citation] = []


def parse_answer(text: str) -> dict:
    """Find the JSON in Claude's reply, then let Pydantic check its structure.
    If anything is missing or the wrong type, return a safe 'not found' result."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        if not match:
            raise ValueError("no JSON object in the reply")
        return AnswerResult.model_validate_json(match.group(0)).model_dump()
    except (ValidationError, ValueError) as err:
        return {"found": False, "answer": text.strip(), "citations": [], "parse_error": str(err)}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def build_prompt(question: str, chunks):
    blocks = [f"[chunk_id={c['id']} | {c['doc']} p{c['page']}]\n{c['text']}" for c in chunks]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def parse_json(text: str):
    """Models sometimes wrap JSON in prose or code fences, so pull out the first {...} block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"found": False, "answer": text.strip(), "citations": [], "parse_error": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"found": False, "answer": text.strip(), "citations": [], "parse_error": True}


def check_citations(result: dict, chunks):
    """Deterministic guardrail: is every quoted passage really in the chunk it cites?
    Returns (n_valid, n_total). This is code checking the model, not another model."""
    by_id = {c["id"]: _norm(c["text"]) for c in chunks}
    cites = result.get("citations") or []
    valid = sum(1 for c in cites
                if c.get("chunk_id") in by_id and _norm(c.get("quote", "")) in by_id[c["chunk_id"]]
                and _norm(c.get("quote", "")))
    return valid, len(cites)


def call_llm(system: str, user: str, max_tokens: int = 2000) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=config.LLM_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    # The reply can contain several blocks (e.g. a "thinking" block before the answer),
    # so keep only the text blocks instead of assuming the first block is text.
    return "".join(b.text for b in msg.content if b.type == "text")


def answer(question: str, llm=call_llm):
    chunks = retrieve(question)
    result = parse_answer(llm(SYSTEM, build_prompt(question, chunks)))
    valid, total = check_citations(result, chunks)
    result["citations_valid"], result["citations_total"] = valid, total
    result["retrieved"] = [{"id": c["id"], "doc": c["doc"], "page": c["page"]} for c in chunks]
    return result, chunks


if __name__ == "__main__":
    import sys
    res, _ = answer(" ".join(sys.argv[1:]))
    print(json.dumps(res, indent=2))
