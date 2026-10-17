"""Tests for PDF writing, retrieval, citations, and graph routing."""

from __future__ import annotations

from pypdf import PdfReader

from rag_agent.citations import parse_sources
from rag_agent.config import get_settings
from rag_agent.graph import message_text, route_after_grade
from rag_agent.ingest import ingest_pdf
from rag_agent.pdf_writer import write_text_pdf
from rag_agent.store import get_vectorstore, reset_client


def test_pdf_round_trip(tmp_path):
    path = tmp_path / "notes.pdf"
    write_text_pdf(
        path,
        [["The assigned berth is Pier 7, berth C."]],
    )
    text = PdfReader(str(path)).pages[0].extract_text()
    assert "Pier 7" in text


def test_parse_sources():
    content = (
        "Source: harbor-ops-brief.pdf | Page: 1\n"
        "The docking window is 04:30 to 06:15 local time.\n"
        "---\n"
        "Source: harbor-ops-brief.pdf | Page: 2\n"
        "The invoice code for this call is NH-4481."
    )
    sources = parse_sources(content)
    assert len(sources) == 2
    assert sources[0].page == 1
    assert "04:30" in sources[0].excerpt
    assert sources[1].source == "harbor-ops-brief.pdf"


def test_message_text_skips_reasoning_blocks():
    class Message:
        content = [
            {"type": "reasoning", "text": "hidden"},
            {"type": "text", "text": "The berth is Pier 7."},
        ]

    assert message_text(Message()) == "The berth is Pier 7."


def test_route_after_grade(monkeypatch):
    monkeypatch.setenv("MAX_REWRITES", "1")
    assert get_settings().max_rewrites == 1
    assert route_after_grade({"grade": "relevant", "rewrites": 0}) == "generate_answer"
    assert route_after_grade({"grade": "empty", "rewrites": 0}) == "generate_answer"
    assert (
        route_after_grade({"grade": "irrelevant", "rewrites": 0}) == "rewrite_question"
    )
    assert (
        route_after_grade({"grade": "irrelevant", "rewrites": 1}) == "generate_answer"
    )


def test_ingest_and_search(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    reset_client()
    path = tmp_path / "notes.pdf"
    write_text_pdf(
        path,
        [
            [
                "The backup berth, if Pier 7 is occupied, is Pier 3, berth A. "
                "The invoice code for this call is NH-4481."
            ]
        ],
    )
    result = ingest_pdf(path)
    assert result.error is None
    assert result.chunks >= 1
    documents = get_vectorstore().similarity_search("backup berth", k=2)
    assert documents
    assert "Pier 3" in documents[0].page_content
    assert documents[0].metadata["source"] == "notes.pdf"
    assert documents[0].metadata["page"] == 1
    reset_client()
