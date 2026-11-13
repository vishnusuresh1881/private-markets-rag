"""Step 5: a Streamlit front end that CALLS THE API (it never imports the rag/ code).

Run (the API must already be running in another terminal):
    export RAG_API_KEY=...          # the same key the API was started with
    streamlit run ui/app.py         # opens http://localhost:8501

Optional: RAG_API_URL (default http://127.0.0.1:8000) to point at an API elsewhere.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("RAG_API_KEY", "")
TIMEOUT_S = 90  # an answer takes a few seconds; never wait forever on a hung server

EXAMPLES = [
    "What is the definition of fair value?",
    "What was 3i's NAV per share at the end of FY2026?",
    "What does ILPA recommend for holding carried interest in escrow?",
    "What is the UK statutory maternity pay rate?",   # should be "not found"
]


def md(text: str) -> str:
    """Streamlit renders markdown, where $...$ becomes maths. Escape it in document text."""
    return text.replace("$", r"\$")


@st.cache_data(ttl=10)
def check_health():
    """GET /health. Cached for 10 s because Streamlit re-runs this whole script on every click."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def ask_api(question: str):
    """POST /ask. Returns (data, None) on success or (None, user-friendly error message)."""
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question},
                          headers={"X-API-Key": API_KEY}, timeout=TIMEOUT_S)
    except requests.ConnectionError:
        return None, f"Can't reach the API at {API_URL}. Is uvicorn running?"
    except requests.Timeout:
        return None, f"The API took longer than {TIMEOUT_S}s. Try again."
    if r.status_code == 200:
        return r.json(), None
    messages = {
        401: "The API rejected the key. Set RAG_API_KEY to the same value the API was started with.",
        422: "The question must be between 3 and 500 characters.",
        503: "The language model is unavailable right now (or the API has no Anthropic key). Try again.",
    }
    return None, messages.get(r.status_code, f"Unexpected error from the API (HTTP {r.status_code}).")


# --- Page ---------------------------------------------------------------------
st.set_page_config(page_title="Private markets RAG", page_icon="📄", layout="wide")
st.title("Private markets document assistant")
st.caption("Answers only from IPEV, ILPA and 3i documents, quotes its sources, "
           "and says when the documents don't answer.")

with st.sidebar:
    st.subheader("API status")
    health = check_health()
    if health:
        st.success(f"Connected · {health['chunks']:,} chunks indexed")
    else:
        st.error(f"API not reachable at {API_URL}")
    if not API_KEY:
        st.warning("RAG_API_KEY is not set in this terminal, so questions will be rejected.")
    st.caption(f"API: {API_URL}")
    st.subheader("Try")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.question = q   # fills the text box on the next run

# A form means typing doesn't trigger a re-run; only the button does.
with st.form("ask"):
    question = st.text_area("Your question", key="question", height=80,
                            placeholder="e.g. What is the definition of fair value?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if len(question.strip()) < 3:
        st.warning("Please type a question first.")
        st.stop()
    with st.spinner("Retrieving passages and asking Claude..."):
        data, error = ask_api(question.strip())
    if error:
        st.error(error)
        st.stop()

    # --- Answer -----------------------------------------------------------------
    if data["found"]:
        st.subheader("Answer")
        st.markdown(md(data["answer"]))
    else:
        st.warning("**Not found in the documents.** " + md(data["answer"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Quotes verified verbatim", f"{data['citations_valid']}/{data['citations_total']}")
    c2.metric("Passages retrieved", len(data["passages"]))
    c3.metric("Latency", f"{data['latency_ms'] / 1000:.1f} s")

    # --- Citations: show each quote with the document and page it came from ---------
    by_id = {p["id"]: p for p in data["passages"]}
    if data["citations"]:
        st.subheader("Sources")
        for c in data["citations"]:
            p = by_id.get(c["chunk_id"])
            where = f"{p['doc']}, page {p['page']}" if p else f"chunk {c['chunk_id']} (not in retrieved passages)"
            st.markdown(f"> {md(c['quote'])}\n\n— *{md(where)}*")

    # --- Retrieved passages: everything the model was shown, best first ------------
    st.subheader("Retrieved passages")
    st.caption("What retrieval sent to the model. Scores are RRF scores in hybrid mode (higher is better).")
    cited = {c["chunk_id"] for c in data["citations"]}
    for rank, p in enumerate(data["passages"], start=1):
        label = f"#{rank} · {p['doc']} p{p['page']} · score {p['score']:.4f}"
        if p["id"] in cited:
            label += " · cited"
        with st.expander(label):
            st.markdown(md(p["text"]))
