"""Download Magnificent 7 earnings figures from SEC EDGAR and store them in Chroma."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import ROOT, get_settings
from rag_agent.store import clear_library, get_vectorstore, reset_client
from rag_agent.streams import revenue_streams

# SEC asks every client to send a descriptive User-Agent.
_USER_AGENT = "Mag7EarningsAgent research@example.com"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
_CACHE = ROOT / "data" / "edgar"
_FILING_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
_HEADER = re.compile(r"<ix:header\b.*?</ix:header>", re.IGNORECASE | re.DOTALL)
_XBRLISH = re.compile(r"xbrli:|fasb\.org|iso4217:|us-gaap:", re.IGNORECASE)
_CONTRACT = re.compile(
    r"\b(contracts?|agreements?|customers?|partners?|partnerships?|suppliers?|"
    r"alliances?|licensees?|licensors?)\b|entered into|material definitive",
    re.IGNORECASE,
)

MAG7: dict[str, tuple[str, str]] = {
    "AAPL": ("0000320193", "Apple"),
    "MSFT": ("0000789019", "Microsoft"),
    "GOOGL": ("0001652044", "Alphabet"),
    "AMZN": ("0001018724", "Amazon"),
    "NVDA": ("0001045810", "NVIDIA"),
    "META": ("0001326801", "Meta Platforms"),
    "TSLA": ("0001318605", "Tesla"),
}

# First matching concept wins for a line, so the preferred tag is listed first.
_LINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Revenue", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")),
    ("Cost of revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    ("Gross profit", ("GrossProfit",)),
    ("Operating income", ("OperatingIncomeLoss",)),
    ("Net income", ("NetIncomeLoss",)),
    ("Basic EPS", ("EarningsPerShareBasic",)),
    ("Diluted EPS", ("EarningsPerShareDiluted",)),
)

_ANNUAL = re.compile(r"^CY\d{4}$")
_QUARTER = re.compile(r"^CY\d{4}Q[1-4]$")
_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}


@dataclass(frozen=True, slots=True)
class EarningsLoad:
    ticker: str
    company: str
    periods: int
    contract_passages: int = 0
    error: str | None = None


def normalize_ticker(value: str) -> str | None:
    ticker = (value or "").strip().upper()
    if ticker == "GOOG":
        ticker = "GOOGL"
    if ticker in MAG7:
        return ticker
    return None


def load_mag7(*, pause_seconds: float = 0.12) -> list[EarningsLoad]:
    """Store two years of earnings, and filing passages for contract questions."""
    cutoff = _cutoff(date.today())
    results: list[EarningsLoad] = []
    documents: list[Document] = []
    ids: list[str] = []
    for ticker, (cik, company) in MAG7.items():
        try:
            payload = fetch_company_facts(cik)
            time.sleep(pause_seconds)
            submissions = fetch_submissions(cik)
            passages, streams = filing_documents(
                ticker,
                company,
                cik,
                submissions,
                cutoff,
                pause_seconds=pause_seconds,
            )
            statements = statement_documents(
                ticker,
                company,
                payload,
                streams=streams,
                filing_urls=_filing_urls(cik, submissions),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            results.append(EarningsLoad(ticker, company, 0, 0, str(exc)))
            time.sleep(pause_seconds)
            continue
        if not statements and not passages:
            results.append(EarningsLoad(ticker, company, 0, 0, "SEC returned nothing for this window."))
        else:
            built = statements + passages
            for index, document in enumerate(built):
                documents.append(document)
                ids.append(f"{ticker}:{index}")
            results.append(
                EarningsLoad(ticker, company, len(statements), len(passages))
            )
        time.sleep(pause_seconds)

    if documents:
        _replace(documents, ids)
    return results


def fetch_company_facts(cik: str) -> dict:
    return _get_json(_FACTS_URL.format(cik=cik))


def fetch_submissions(cik: str) -> dict:
    return _get_json(_SUBMISSIONS_URL.format(cik=cik))


def normalize_focus(value: str) -> str:
    """Map a tool argument to latest_quarter, latest_year, or contracts."""
    text = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"latest_year", "annual", "year", "fiscal_year", "latest_annual"}:
        return "latest_year"
    if text in {"contracts", "contract", "customers", "partners", "agreements"}:
        return "contracts"
    return "latest_quarter"


def statement_documents(
    ticker: str,
    company: str,
    payload: dict,
    *,
    as_of: date | None = None,
    streams: dict[tuple[str, str], list[tuple[str, float]]] | None = None,
    filing_urls: dict[str, str] | None = None,
) -> list[Document]:
    """Keep every earnings period in the last two years, and mark the latest ones."""
    cutoff = _cutoff(as_of or date.today())
    periods = [
        item
        for item in _periods(payload)
        if item["end"] >= cutoff or item["filed"] >= cutoff
    ]
    latest_quarter = max((item["end"] for item in periods if item["kind"] == "quarter"), default="")
    latest_year = max((item["end"] for item in periods if item["kind"] == "annual"), default="")
    documents: list[Document] = []
    for period in periods:
        if period["kind"] == "quarter" and period["end"] == latest_quarter:
            focus = "latest_quarter"
        elif period["kind"] == "annual" and period["end"] == latest_year:
            focus = "latest_year"
        else:
            focus = "history"
        text = _render(
            ticker,
            company,
            period,
            focus,
            (streams or {}).get((period["end"], period["kind"]), []),
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"{ticker} {period['label']}",
                    "ticker": ticker,
                    "company": company,
                    "period": period["label"],
                    "form": period["form"],
                    "focus_tag": focus,
                    "page": 1,
                    "url": (filing_urls or {}).get(str(period.get("accn") or ""), ""),
                },
            )
        )
    if not documents:
        return []
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def _periods(payload: dict) -> list[dict]:
    facts = ((payload.get("facts") or {}).get("us-gaap")) or {}
    grouped: dict[str, dict] = {}
    for label, concepts in _LINES:
        for concept in concepts:
            fact = facts.get(concept) or {}
            for row in _framed_rows(fact):
                frame = str(row["frame"])
                bucket = grouped.setdefault(
                    frame,
                    {
                        "frame": frame,
                        "kind": "annual" if _ANNUAL.match(frame) else "quarter",
                        "end": str(row.get("end") or ""),
                        "filed": str(row.get("filed") or ""),
                        "form": str(row.get("form") or ""),
                        "fy": row.get("fy"),
                        "fp": str(row.get("fp") or ""),
                        "accn": str(row.get("accn") or ""),
                        "lines": {},
                    },
                )
                if label in bucket["lines"]:
                    continue
                filed = str(row.get("filed") or "")
                if filed >= bucket["filed"]:
                    bucket["filed"] = filed
                    bucket["accn"] = str(row.get("accn") or bucket.get("accn") or "")
                    bucket["end"] = str(row.get("end") or bucket["end"])
                    bucket["form"] = str(row.get("form") or bucket["form"])
                    bucket["fy"] = row.get("fy", bucket["fy"])
                    bucket["fp"] = str(row.get("fp") or bucket["fp"])
                bucket["lines"][label] = (row.get("val"), str(row.get("unit") or ""))
    periods = [item for item in grouped.values() if item["lines"]]
    periods.sort(key=lambda item: item["end"], reverse=True)
    return periods


def _framed_rows(fact: dict) -> list[dict]:
    rows: list[dict] = []
    for unit, items in (fact.get("units") or {}).items():
        if unit not in {"USD", "USD/shares"}:
            continue
        for item in items:
            frame = str(item.get("frame") or "")
            if item.get("form") not in _FORMS:
                continue
            if not (_ANNUAL.match(frame) or _QUARTER.match(frame)):
                continue
            rows.append({**item, "unit": unit})
    return rows


def _render(
    ticker: str,
    company: str,
    period: dict,
    focus: str,
    streams: list[tuple[str, float]] | None = None,
) -> str:
    period["label"] = _label(period)
    if focus == "latest_quarter":
        standing = "This is the latest quarterly earnings statement."
    elif focus == "latest_year":
        standing = "This is the latest annual earnings statement."
    else:
        standing = "This is an earlier earnings statement from the last two years."
    lines = [
        f"Ticker: {ticker}",
        f"Company: {company}",
        f"Period: {period['label']}",
        f"Period end: {period['end']}",
        f"Filed: {period['filed']}",
        f"Form: {period['form']}",
        f"Fiscal year: {period['fy']} {period['fp']}".strip(),
        standing,
        "Figures are from SEC EDGAR XBRL company facts, as reported in the filing.",
    ]
    for label, _concepts in _LINES:
        value = period["lines"].get(label)
        if not value or value[0] is None:
            continue
        lines.append(f"{label}: {_format_value(value[0], value[1])}")
    if streams:
        lines.append("Revenue streams:")
        for label, value in streams:
            lines.append(f"{label}: {_format_value(value, 'USD')}")
    return "\n".join(lines)


def _label(period: dict) -> str:
    frame = period["frame"]
    match = _QUARTER.match(frame)
    if match:
        return f"{frame[2:6]} {frame[-2:]} ending {period['end']}"
    return f"Fiscal year ending {period['end']}"


def _format_value(value: object, unit: str) -> str:
    number = float(value)
    if unit == "USD/shares":
        return f"${number:,.2f} per share"
    if unit == "USD":
        sign = "-" if number < 0 else ""
        return f"{sign}${abs(number):,.0f}"
    return f"{number:,.4g}"


def filing_documents(
    ticker: str,
    company: str,
    cik: str,
    submissions: dict,
    cutoff: str,
    *,
    pause_seconds: float = 0.12,
) -> tuple[list[Document], dict[tuple[str, str], list[tuple[str, float]]]]:
    """Index contract passages and collect revenue streams from the same filings."""
    documents: list[Document] = []
    streams: dict[tuple[str, str], list[tuple[str, float]]] = {}
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    for filing in _recent_filings(submissions, cutoff):
        try:
            html = _download_filing(ticker, cik, filing)
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(pause_seconds)
            continue
        time.sleep(pause_seconds)
        _keep_streams(streams, revenue_streams(html))
        passages = contract_passages(_html_to_text(html), form=filing["form"])
        for passage in passages:
            documents.append(
                Document(
                    page_content=(
                        f"Ticker: {ticker}\nCompany: {company}\n"
                        f"Form: {filing['form']}\nFiled: {filing['filed']}\n\n{passage}"
                    ),
                    metadata={
                        "source": f"{ticker} {filing['form']} filed {filing['filed']}",
                        "ticker": ticker,
                        "company": company,
                        "period": filing["filed"],
                        "form": filing["form"],
                        "focus_tag": "contracts",
                        "page": 1,
                        "url": _archive_url(cik, filing["accession"], filing["document"]),
                    },
                )
            )
    if not documents:
        return [], streams
    return splitter.split_documents(documents), streams


def _keep_streams(
    chosen: dict[tuple[str, str], list[tuple[str, float]]],
    found: list,
) -> None:
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for item in found:
        grouped.setdefault((item.end, item.kind), []).append((item.label, item.value))
    for key, rows in grouped.items():
        current = chosen.get(key)
        if current is None or len(rows) > len(current):
            chosen[key] = rows


def contract_passages(text: str, *, form: str) -> list[str]:
    """Keep paragraphs that name customers, suppliers, or agreements."""
    agreement_filing = form.startswith("8-K")
    kept: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if len(cleaned) < 80 or _XBRLISH.search(cleaned):
            continue
        if not agreement_filing and not _CONTRACT.search(cleaned):
            continue
        if len(cleaned) > 1800:
            cleaned = cleaned[:1800].rsplit(" ", 1)[0]
        kept.append(cleaned)
        if len(kept) >= 40:
            break
    return kept


def _recent_filings(submissions: dict, cutoff: str) -> list[dict]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    items = recent.get("items") or []
    chosen: list[dict] = []
    for index, form in enumerate(forms):
        filed = dates[index] if index < len(dates) else ""
        if filed < cutoff:
            continue
        item = items[index] if index < len(items) else ""
        if form not in _FILING_FORMS and not (form in {"8-K", "8-K/A"} and "1.01" in (item or "")):
            continue
        document = documents[index] if index < len(documents) else ""
        accession = accessions[index] if index < len(accessions) else ""
        if not document or not accession:
            continue
        chosen.append({"form": form, "filed": filed, "accession": accession, "document": document})
    return chosen


def _filing_urls(cik: str, submissions: dict) -> dict[str, str]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    urls: dict[str, str] = {}
    for index, accession in enumerate(accessions):
        document = documents[index] if index < len(documents) else ""
        if accession and document:
            urls[str(accession)] = _archive_url(cik, str(accession), str(document))
    return urls


def _archive_url(cik: str, accession: str, document: str) -> str:
    if not accession or not document:
        return ""
    return _ARCHIVE.format(
        cik=str(int(cik)),
        accession=accession.replace("-", ""),
        document=document,
    )


def _download_filing(ticker: str, cik: str, filing: dict) -> str:
    folder = _CACHE / ticker
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{filing['accession']}.html"
    if path.is_file() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace")
    url = _archive_url(cik, filing["accession"], filing["document"])
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    return text


def _html_to_text(html: str) -> str:
    cleaned = _HEADER.sub(" ", html)
    parser = _TextExtractor()
    parser.feed(cleaned)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n\n", text)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        if tag in {"p", "div", "br", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _cutoff(day: date) -> str:
    try:
        start = day.replace(year=day.year - 2)
    except ValueError:
        start = day.replace(year=day.year - 2, day=28)
    return start.isoformat()


def _get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _replace(documents: list[Document], ids: list[str]) -> None:
    clear_library()
    store = get_vectorstore()
    for start in range(0, len(documents), 64):
        store.add_documents(documents[start : start + 64], ids=ids[start : start + 64])


def main() -> None:
    reset_client()
    results = load_mag7()
    for result in results:
        if result.error:
            print(f"{result.ticker}: {result.error}")
        else:
            print(
                f"{result.ticker}: stored {result.periods} earnings periods "
                f"and {result.contract_passages} contract passages"
            )


if __name__ == "__main__":
    main()
