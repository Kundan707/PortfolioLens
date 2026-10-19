"""Split filing text into chunks, by fixed size or by shifts in meaning between sentences."""

from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import get_settings
from rag_agent.store import get_embeddings

_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"(“])")
# bge-small reads at most 512 tokens, so a longer chunk would be cut off when embedded.
_MAX_CHARS = 1800
_MIN_CHARS = 250
_BREAK_PERCENTILE = 90


def fixed_chunks(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def semantic_chunks(header: str, paragraphs: list[str], metadata: dict) -> list[Document]:
    """Group consecutive sentences until the meaning shifts, then start a new chunk.

    A break goes where the embedding distance between neighbouring sentences is in the top
    tenth for this filing. Each chunk keeps the filing header so it reads on its own.
    """
    sentences = _sentences(paragraphs)
    if not sentences:
        return []
    groups = _group(sentences, _breaks(sentences))
    return [
        Document(page_content=f"{header}\n\n{text}", metadata=dict(metadata))
        for text in groups
    ]


def _sentences(paragraphs: list[str]) -> list[str]:
    found: list[str] = []
    for paragraph in paragraphs:
        for sentence in _SENTENCE.split(paragraph):
            sentence = sentence.strip()
            while len(sentence) > _MAX_CHARS:
                cut = sentence[:_MAX_CHARS].rsplit(" ", 1)[0] or sentence[:_MAX_CHARS]
                found.append(cut)
                sentence = sentence[len(cut) :].strip()
            if sentence:
                found.append(sentence)
    return found


def _breaks(sentences: list[str]) -> set[int]:
    if len(sentences) < 3:
        return set()
    # Each sentence is embedded with its neighbours so one short line does not swing the signal.
    windows = [" ".join(sentences[max(0, i - 1) : i + 2]) for i in range(len(sentences))]
    vectors = np.array(get_embeddings().embed_documents(windows), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = float(np.percentile(distances, _BREAK_PERCENTILE))
    return {index + 1 for index, distance in enumerate(distances) if distance >= threshold}


def _group(sentences: list[str], breaks: set[int]) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for index, sentence in enumerate(sentences):
        at_break = index in breaks and size >= _MIN_CHARS
        if current and (at_break or size + len(sentence) + 1 > _MAX_CHARS):
            groups.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        tail = " ".join(current)
        if groups and len(tail) < _MIN_CHARS and len(groups[-1]) + len(tail) + 1 <= _MAX_CHARS:
            groups[-1] = f"{groups[-1]} {tail}"
        else:
            groups.append(tail)
    return groups
