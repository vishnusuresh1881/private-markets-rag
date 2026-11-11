"""Day 2: score the assistant on a fixed test set.

Run:
  python -m eval.run_eval --label baseline           # full run (needs ANTHROPIC_API_KEY)
  python -m eval.run_eval --no-llm --label retrieval # retrieval only, free and fast

Each test case has expected_phrases: short strings that must appear in the retrieved text.
Using phrases (not chunk ids) means your test set still works after you change chunk size.
"""
import argparse
import datetime
import json
import pathlib
import re
from rag import config
from rag.retrieve import retrieve
from rag.answer import answer, call_llm, parse_json, _norm

JUDGE_SYSTEM = """You grade an assistant's answer against a reference answer.
Verdict "correct": same key facts as the reference, nothing contradicting it.
Verdict "partial": some key facts right but something important missing or wrong.
Verdict "incorrect": wrong, missing the main point, or says it cannot find the answer.
Reply with JSON only: {"verdict": "correct|partial|incorrect", "reason": "one sentence"}"""


def retrieval_hit(case, chunks) -> bool:
    """True if EVERY expected phrase appears in at least one retrieved chunk."""
    text = " ".join(_norm(c["text"]) for c in chunks)
    return all(_norm(p) in text for p in case["expected_phrases"])


def judge(case, model_answer: str, llm) -> dict:
    prompt = (f"Question: {case['question']}\nReference answer: {case['expected_answer']}\n"
              f"Assistant answer: {model_answer}")
    out = parse_json(llm(JUDGE_SYSTEM, prompt, 1000))
    return {"verdict": out.get("verdict", "incorrect"), "reason": out.get("reason", "")}


def run(test_path, label, no_llm=False, llm=call_llm):
    cases = json.loads(pathlib.Path(test_path).read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        row = {"id": case["id"], "type": case["type"], "question": case["question"]}
        if no_llm:
            chunks = retrieve(case["question"])
            row["retrieval_hit"] = retrieval_hit(case, chunks) if case["type"] != "unanswerable" else None
        else:
            res, chunks = answer(case["question"], llm=llm)
            row["retrieval_hit"] = retrieval_hit(case, chunks) if case["type"] != "unanswerable" else None
            row["found"] = bool(res.get("found"))
            row["answer"] = res.get("answer", "")
            row["citations_valid"] = res["citations_valid"]
            row["citations_total"] = res["citations_total"]
            if case["type"] == "unanswerable":
                row["refused_correctly"] = not row["found"]
            else:
                j = judge(case, row["answer"], llm)
                row["verdict"], row["judge_reason"] = j["verdict"], j["reason"]
        rows.append(row)
    summary = summarise(rows, no_llm)
    out = {"label": label, "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "config": {k: getattr(config, k) for k in
                      ["CHUNK_WORDS", "CHUNK_OVERLAP_WORDS", "TOP_K", "RETRIEVAL_MODE", "LLM_MODEL"]},
           "summary": summary, "rows": rows}
    pathlib.Path("results").mkdir(exist_ok=True)
    path = pathlib.Path("results") / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{label}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    report(out, path)
    return out


def pct(n, d):
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"


def summarise(rows, no_llm):
    ans = [r for r in rows if r["type"] != "unanswerable"]
    s = {"retrieval_hit_rate": pct(sum(1 for r in ans if r["retrieval_hit"]), len(ans))}
    if no_llm:
        return s
    unans = [r for r in rows if r["type"] == "unanswerable"]
    cites = sum(r["citations_total"] for r in rows)
    s["answer_correct"] = pct(sum(1 for r in ans if r["verdict"] == "correct"), len(ans))
    s["answer_partial_or_better"] = pct(sum(1 for r in ans if r["verdict"] in ("correct", "partial")), len(ans))
    s["refusal_accuracy"] = pct(sum(1 for r in unans if r["refused_correctly"]), len(unans))
    s["citations_verbatim"] = pct(sum(r["citations_valid"] for r in rows), cites)
    return s


def report(out, path):
    print(f"\n=== {out['label']}  {out['config']} ===")
    for k, v in out["summary"].items():
        print(f"  {k:26s} {v}")
    fails = [r for r in out["rows"]
             if r.get("retrieval_hit") is False or r.get("verdict") in ("partial", "incorrect")
             or r.get("refused_correctly") is False]
    if fails:
        print("\nFailures to analyse (these are your interview stories):")
        for r in fails:
            why = []
            if r.get("retrieval_hit") is False: why.append("retrieval missed")
            if r.get("verdict") in ("partial", "incorrect"): why.append(f"answer {r['verdict']}")
            if r.get("refused_correctly") is False: why.append("should have refused")
            print(f"  {r['id']}: {', '.join(why)} - {r['question'][:80]}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="eval/test_set.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-llm", action="store_true", help="retrieval metrics only")
    a = ap.parse_args()
    run(a.test_set, a.label, no_llm=a.no_llm)
