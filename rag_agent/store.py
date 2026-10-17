"""ChromaDB persistence for PDF passages."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_agent.config import COLLECTION_NAME, EMBED_MODEL, ROOT, chroma_path

_SPACE = {"hnsw:space": "cosine"}


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    source: str
    chunks: int
    pages: tuple[int, ...]


class LocalEmbeddings(Embeddings):
    """BGE embeddings through FastEmbed. Documents and queries use matching prefixes."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.query_embed(text)))
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    from fastembed import TextEmbedding

    cache = ROOT / ".cache" / "fastembed"
    cache.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(cache))
    return LocalEmbeddings(model)


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(chroma_path()),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_vectorstore() -> Chroma:
    kwargs: dict = {
        "client": get_client(),
        "collection_name": COLLECTION_NAME,
        "collection_metadata": _SPACE,
    }
    embeddings = get_embeddings()
    parameters = inspect.signature(Chroma.__init__).parameters
    if "embedding" in parameters:
        kwargs["embedding"] = embeddings
    else:
        kwargs["embedding_function"] = embeddings
    return Chroma(**kwargs)


def _open_collection():
    """Return the collection without loading the embedding model."""
    client = get_client()
    names = set()
    for item in client.list_collections():
        names.add(item.name if hasattr(item, "name") else str(item))
    if COLLECTION_NAME not in names:
        return None
    return client.get_collection(COLLECTION_NAME)


def library_count() -> int:
    collection = _open_collection()
    if collection is None:
        return 0
    return int(collection.count())


def list_documents() -> list[IndexedDocument]:
    collection = _open_collection()
    if collection is None or collection.count() == 0:
        return []
    raw = collection.get(include=["metadatas"])
    grouped: dict[str, set[int]] = {}
    counts: dict[str, int] = {}
    for meta in raw.get("metadatas") or []:
        meta = meta or {}
        source = str(meta.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
        grouped.setdefault(source, set())
        page = meta.get("page")
        if isinstance(page, int):
            grouped[source].add(page)
    documents = [
        IndexedDocument(
            source=source,
            chunks=counts[source],
            pages=tuple(sorted(grouped[source])),
        )
        for source in sorted(counts)
    ]
    return documents


def remove_document(source: str) -> int:
    collection = _open_collection()
    if collection is None:
        return 0
    raw = collection.get(where={"source": source})
    ids = list(raw.get("ids") or [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def clear_library() -> None:
    client = get_client()
    existing = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)


def reset_client() -> None:
    """Drop cached clients so tests can point Chroma at a fresh directory."""
    get_client.cache_clear()
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        return
