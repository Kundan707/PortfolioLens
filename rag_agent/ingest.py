"""Read PDFs, split them, and store the passages in ChromaDB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_agent.config import DATA_DIR, get_settings
from rag_agent.store import get_vectorstore, remove_document


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: str
    pages: int
    chunks: int
    error: str | None = None


def ingest_pdf(path: Path) -> IngestResult:
    """Replace any earlier passages from this file, then index the new text."""
    path = path.expanduser().resolve()
    name = path.name
    if path.suffix.lower() != ".pdf":
        return IngestResult(name, 0, 0, "Only PDF files can be indexed.")
    if not path.is_file():
        return IngestResult(name, 0, 0, "File not found.")
    if path.read_bytes()[:5] != b"%PDF-":
        return IngestResult(name, 0, 0, "This file is not a PDF.")

    try:
        documents = _load_pdf(path)
    except Exception as exc:
        return IngestResult(name, 0, 0, f"Could not read this PDF: {exc}")

    for document in documents:
        document.metadata["source"] = name
        page = document.metadata.get("page", 0)
        document.metadata["page"] = int(page) + 1
        document.page_content = document.page_content or ""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]
    if not chunks:
        return IngestResult(
            name,
            len(documents),
            0,
            "No extractable text. Scanned pages need OCR, which this app does not run.",
        )

    remove_document(name)
    store = get_vectorstore()
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{name}:{index}" for index in range(start, start + len(batch))]
        store.add_documents(batch, ids=ids)
    return IngestResult(name, len(documents), len(chunks))


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    documents: list[Document] = []
    for index, page in enumerate(reader.pages):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": path.name, "page": index},
            )
        )
    return documents


def ingest_paths(paths: list[Path]) -> list[IngestResult]:
    results: list[IngestResult] = []
    for path in paths:
        if path.is_dir():
            pdfs = sorted(path.glob("*.pdf"))
            if not pdfs:
                results.append(IngestResult(path.name, 0, 0, "No PDF files in this folder."))
            results.extend(ingest_pdf(pdf) for pdf in pdfs)
        else:
            results.append(ingest_pdf(path))
    return results


def save_upload(filename: str, data: bytes) -> Path:
    """Store an uploaded PDF under data/pdfs using only its file name."""
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".pdf") or name.startswith("."):
        raise ValueError("Upload a PDF file with a plain file name.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / name
    destination.write_bytes(data)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index PDF files into ChromaDB.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF files or folders. Defaults to sample_docs.",
    )
    args = parser.parse_args(argv)
    raw_paths = args.paths or ["sample_docs"]
    paths = [_resolve(value) for value in raw_paths]
    results = ingest_paths(paths)
    failed = False
    for result in results:
        if result.error:
            failed = True
            print(f"{result.source}: {result.error}")
        else:
            print(
                f"{result.source}: {result.pages} page(s), {result.chunks} passage(s)"
            )
    return 1 if failed else 0


def _resolve(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate
    from rag_agent.config import ROOT

    alternative = ROOT / value
    if alternative.exists():
        return alternative
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
