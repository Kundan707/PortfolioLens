"""Paths and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pdfs"
SAMPLE_DIR = ROOT / "sample_docs"
SAMPLE_PDF = SAMPLE_DIR / "harbor-ops-brief.pdf"
COLLECTION_NAME = "pdf_documents"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_rewrites: int


def get_settings() -> Settings:
    """Read settings from the environment. An existing variable wins over `.env`."""
    load_dotenv(ROOT / ".env")
    return Settings(
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        top_k=_positive_int("TOP_K", 4),
        chunk_size=_positive_int("CHUNK_SIZE", 1000),
        chunk_overlap=_positive_int("CHUNK_OVERLAP", 150),
        max_rewrites=_non_negative_int("MAX_REWRITES", 1),
    )


def chroma_path() -> Path:
    override = os.environ.get("CHROMA_DIR")
    path = Path(override) if override else ROOT / "chroma_db"
    path.mkdir(parents=True, exist_ok=True)
    return path


def api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value
