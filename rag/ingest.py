"""Day 1, step 1: read documents, split into overlapping chunks, embed, save.

Run:  python -m rag.ingest
"""
import json
import pathlib
import numpy as np
from pypdf import PdfReader
from . import config


def read_pages(path: pathlib.Path):
    """Yield (page_number, text) for a PDF, or one page for .txt/.md files."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            yield i, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_words(text: str, size: int, overlap: int):
    """Split text into chunks of `size` words, each sharing `overlap` words with the previous one.
    Why overlap? A fact that sits on a chunk boundary would otherwise be cut in half."""
    words = text.split()
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_chunks(docs_dir: str):
    chunks = []
    files = sorted(p for p in pathlib.Path(docs_dir).iterdir()
                   if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise SystemExit(f"No documents found in {docs_dir}/ - add some PDFs first.")
    for path in files:
        empty_pages = 0
        for page_no, text in read_pages(path):
            if not text.strip():
                empty_pages += 1
                continue
            for piece in chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP_WORDS):
                chunks.append({
                    "id": len(chunks),
                    "doc": path.name,
                    "page": page_no,
                    "text": piece,
                })
        if empty_pages:
            print(f"  warning: {path.name} has {empty_pages} page(s) with no extractable text "
                  f"(scanned? you may need OCR)")
    return chunks


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    # normalised vectors mean cosine similarity == dot product
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def main(skip_embeddings: bool = False):
    out = pathlib.Path(config.INDEX_DIR)
    out.mkdir(exist_ok=True)
    chunks = build_chunks(config.DOCS_DIR)
    (out / "chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")
    print(f"{len(chunks)} chunks from {len({c['doc'] for c in chunks})} document(s)")
    if skip_embeddings:
        print("Skipped embeddings (BM25 only).")
        return
    vectors = embed_texts([c["text"] for c in chunks])
    np.save(out / "embeddings.npy", np.asarray(vectors))
    print(f"Saved embeddings: {np.asarray(vectors).shape}")


if __name__ == "__main__":
    import sys
    main(skip_embeddings="--bm25-only" in sys.argv)
