"""Tests for PDF writing, retrieval, citations, and graph routing."""

from __future__ import annotations

from pypdf import PdfReader

from rag_agent.citations import format_passages, parse_sources
from rag_agent.news import headlines_from_rss, select_relevant
from datetime import date

from rag_agent.earnings import contract_passages, normalize_focus, normalize_ticker, statement_documents
from rag_agent.streams import revenue_streams
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
    assert (
        route_after_grade({"grade": "relevant", "rewrites": 0, "focus": "contracts"})
        == "fetch_news"
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


def test_earnings_statements_keep_ticker_and_figures():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 1000,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-01",
                                "frame": "CY2024",
                            },
                            {
                                "end": "2024-09-30",
                                "val": 250,
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2024-10-30",
                                "frame": "CY2024Q3",
                            },
                            {
                                "end": "2020-12-31",
                                "val": 10,
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-02-01",
                                "frame": "CY2020",
                            },
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2024-12-31",
                                "val": 1.5,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-01",
                                "frame": "CY2024",
                            }
                        ]
                    }
                },
            }
        }
    }
    documents = statement_documents("NVDA", "NVIDIA", payload)
    text = "\n".join(document.page_content for document in documents)
    assert normalize_ticker("goog") == "GOOGL"
    assert any(document.metadata["ticker"] == "NVDA" for document in documents)
    assert "$1,000" in text
    assert "$1.50 per share" in text
    assert "2024 Q3 ending 2024-09-30" in text
    assert "2020" not in text
    tags = {document.metadata["focus_tag"] for document in documents}
    assert tags == {"latest_year", "latest_quarter"}
    assert normalize_focus("contracts") == "contracts"
    assert normalize_focus("fiscal year") == "latest_year"
    passages = contract_passages(
        "The company entered into a supply agreement with Acme Robotics during the quarter.\n\n"
        "Revenue increased.\n\n"
        "xbrli:sharesiso4217:USD customer",
        form="10-Q",
    )
    assert passages == [
        "The company entered into a supply agreement with Acme Robotics during the quarter."
    ]


def test_news_keeps_headlines_from_the_last_year():
    xml_text = """<?xml version="1.0"?>
    <rss><channel>
    <item>
      <title>OpenAI signs a chip deal - Reuters</title>
      <link>https://www.reuters.com/openai-chips</link>
      <pubDate>Wed, 01 Apr 2026 12:00:00 GMT</pubDate>
      <description>OpenAI agreed to buy chips.</description>
    </item>
    <item>
      <title>OpenAI redesigns its homepage - The Verge</title>
      <link>https://www.theverge.com/openai-homepage</link>
      <pubDate>Thu, 02 Apr 2026 12:00:00 GMT</pubDate>
      <description>The company updated its website.</description>
    </item>
    <item>
      <title>Old story - Reuters</title>
      <pubDate>Wed, 01 Jan 2020 12:00:00 GMT</pubDate>
      <description>This is too old to summarize.</description>
    </item>
    </channel></rss>"""
    items = headlines_from_rss(xml_text, as_of=date(2026, 10, 17))
    assert len(items) == 2
    assert items[0].title == "OpenAI signs a chip deal"
    assert items[0].publisher == "Reuters"
    assert items[0].published == "2026-04-01"
    assert items[0].url == "https://www.reuters.com/openai-chips"
    relevant = select_relevant(
        "OpenAI",
        items,
        "Microsoft entered into an agreement with OpenAI.",
    )
    assert [item.url for item in relevant] == ["https://www.reuters.com/openai-chips"]
    parsed = parse_sources(
        "Source: Reuters — OpenAI — 2026-04-01 | Page: 1\n"
        "Link: https://www.reuters.com/openai-chips\n"
        "OpenAI signs a chip deal"
    )
    assert parsed[0].url == "https://www.reuters.com/openai-chips"
    assert "Link:" not in parsed[0].excerpt


def test_filing_passage_keeps_the_document_link():
    class Passage:
        metadata = {
            "source": "AAPL Fiscal year ending 2025-09-27",
            "page": 1,
            "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        }
        page_content = "Revenue: $416,161,000,000"

    parsed = parse_sources(format_passages([Passage()]))
    assert parsed[0].source.startswith("AAPL")
    assert parsed[0].url.startswith("https://www.sec.gov/")
    assert "Revenue" in parsed[0].excerpt


def test_revenue_streams_from_filing_facts():
    html = """
    <xbrli:context id="total">
      <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
    </xbrli:context>
    <xbrli:context id="phone">
      <xbrli:segment><xbrldi:explicitMember dimension="srt:ProductOrServiceAxis">aapl:IPhoneMember</xbrldi:explicitMember></xbrli:segment>
      <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
    </xbrli:context>
    <xbrli:context id="services">
      <xbrli:segment><xbrldi:explicitMember dimension="srt:ProductOrServiceAxis">us-gaap:ServiceMember</xbrldi:explicitMember></xbrli:segment>
      <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
    </xbrli:context>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="total" scale="6">100</ix:nonFraction>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="phone" scale="6">70</ix:nonFraction>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="services" scale="6">30</ix:nonFraction>
    """
    rows = {(item.label, item.value) for item in revenue_streams(html)}
    assert ("iPhone", 70_000_000) in rows
    assert ("Services", 30_000_000) in rows
