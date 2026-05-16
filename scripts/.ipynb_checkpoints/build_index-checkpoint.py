from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from scripts.config import (
    BM25_FILE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNKS_FILE,
    EMBED_DIR,
    EMBEDDING_MODEL,
    META_FILE,
    bm25_tokenize,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_bm25(chunks: list[dict[str, Any]]) -> BM25Okapi:
    """Build BM25 index using the shared canonical tokeniser."""
    tokenized_corpus = [bm25_tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def build_chroma(chunks: list[dict[str, Any]]) -> Chroma:
    """Build the dense Chroma index.

    We store page_content with the `passage:` prefix because the retriever
    queries with `query:` for E5-style embedding alignment.
    """
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    documents: list[Document] = []
    ids: list[str] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        source = chunk["source"]
        page = chunk["page"]
        title = chunk["title"]
        section = chunk.get("section")
        text = chunk["text"]
        tokens_est = chunk["tokens_est"]
        chunking_method = chunk.get("chunking_method", "unknown")

        doc = Document(
            page_content=f"passage: {text}",
            metadata={
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "title": title,
                "section": section,
                "tokens_est": tokens_est,
                "chunking_method": chunking_method,
            },
        )
        documents.append(doc)
        ids.append(chunk_id)

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    existing = vectorstore.get()
    existing_ids = existing.get("ids", []) if existing else []

    if existing_ids:
        try:
            vectorstore.delete(ids=existing_ids)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to clear existing Chroma collection before rebuild: {exc}"
            ) from exc

    vectorstore.add_documents(documents=documents, ids=ids)
    return vectorstore


def save_bm25(bm25: BM25Okapi, chunks: list[dict[str, Any]]) -> None:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_FILE.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def save_metadata(chunks: list[dict[str, Any]]) -> None:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "title": chunk["title"],
            "section": chunk.get("section"),
            "tokens_est": chunk["tokens_est"],
            "chunking_method": chunk.get("chunking_method", "unknown"),
        }
        for chunk in chunks
    ]

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    required_fields = {"chunk_id", "source", "page", "title", "text", "tokens_est"}
    seen_ids: set[str] = set()

    for i, chunk in enumerate(chunks):
        missing = required_fields - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk #{i} is missing required fields: {sorted(missing)}")

        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")
        seen_ids.add(chunk_id)

        if not str(chunk["text"]).strip():
            raise ValueError(f"Chunk {chunk_id} has empty text")


def main() -> None:
    chunks = load_jsonl(CHUNKS_FILE)
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_FILE}")

    validate_chunks(chunks)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)
    save_bm25(bm25, chunks)
    print(f"Saved BM25 index → {BM25_FILE}")

    print("Building Chroma dense index...")
    build_chroma(chunks)
    print(f"Saved Chroma collection → {CHROMA_DIR}")

    save_metadata(chunks)
    print(f"Saved chunk metadata → {META_FILE}")

    print(f"\nDone. Indexed {len(chunks)} chunks.")
    print(f"Embedding model    : {EMBEDDING_MODEL}")
    print(f"Chroma collection  : {CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()