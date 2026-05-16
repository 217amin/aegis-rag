from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    CHUNKING_METHOD,
    PDF_DIR,
)


def clean_text(text: str) -> str:
    """Basic PDF cleanup while preserving useful retrieval signals."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]+\n", "\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            cleaned = clean_text(page.extract_text() or "")
            if cleaned:
                pages.append({"source": pdf_path.name, "page": page_num, "text": cleaned})
    return pages


def recursive_chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def semantic_chunk_text(text: str) -> list[str]:
    """
    Lightweight semantic-ish chunking.  Groups paragraphs into coherent
    blocks; falls back to recursive splitting for oversized paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= CHUNK_SIZE:
                current = para
            else:
                chunks.extend(recursive_chunk_text(para))
                current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str) -> list[str]:
    if CHUNKING_METHOD == "recursive":
        return recursive_chunk_text(text)
    if CHUNKING_METHOD == "semantic":
        return semantic_chunk_text(text)
    raise ValueError(f"Unsupported CHUNKING_METHOD: {CHUNKING_METHOD}")


def extract_title_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def infer_section_label(text: str) -> str | None:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:6]:
        if 4 <= len(line.split()) <= 12 and not line.endswith(":") and line == line.title():
            return line
    return None


def build_chunks_from_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page_item in pages:
        source, page, text = page_item["source"], page_item["page"], page_item["text"]
        section = infer_section_label(text)
        for i, chunk in enumerate(chunk_text(text), start=1):
            chunk_id = f"{Path(source).stem}_p{page}_c{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "title": extract_title_from_filename(source),
                "section": section,
                "text": chunk,
                "tokens_est": estimate_tokens(chunk),
                "chunking_method": CHUNKING_METHOD,
            })
    return chunks


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    all_pages: list[dict[str, Any]] = []
    for pdf_file in pdf_files:
        all_pages.extend(extract_pdf_pages(pdf_file))

    all_chunks = build_chunks_from_pages(all_pages)
    save_jsonl(all_chunks, CHUNKS_FILE)

    print(f"Processed PDFs : {len(pdf_files)}")
    print(f"Extracted pages: {len(all_pages)}")
    print(f"Saved chunks   : {len(all_chunks)}")
    print(f"Output file    : {CHUNKS_FILE}")
    print(f"Chunking method: {CHUNKING_METHOD}")


if __name__ == "__main__":
    main()
