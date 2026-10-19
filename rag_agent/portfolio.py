"""Read the holdings in the user's brokerage portfolio download."""

from __future__ import annotations

import csv
import io
import re
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from rag_agent.config import portfolio_path

_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_NOT_HOLDINGS = {"CASH", "TOTAL", "SYMBOL"}
_ALIASES = {"GOOG": "GOOGL", "BRK.B": "BRK-B", "BRK/B": "BRK-B"}


class PortfolioError(RuntimeError):
    pass


def clean_ticker(value: str) -> str:
    ticker = (value or "").strip().upper()
    return _ALIASES.get(ticker, ticker)


def portfolio_tickers(path: Path | None = None) -> tuple[str, ...]:
    """Return the symbols held in the portfolio file, in file order."""
    target = Path(path) if path else portfolio_path()
    if not target.is_file():
        raise PortfolioError(f"No portfolio file was found at {target}.")
    return _cached_tickers(str(target), target.stat().st_mtime_ns)


def in_portfolio(ticker: str, path: Path | None = None) -> bool:
    return clean_ticker(ticker) in portfolio_tickers(path)


def tickers_from_rows(rows: list[list]) -> tuple[str, ...]:
    column = 0
    start = 0
    for index, row in enumerate(rows):
        cells = [str(cell or "").strip().lower() for cell in row]
        if "symbol" in cells:
            column = cells.index("symbol")
            start = index + 1
            break
    found: list[str] = []
    for row in rows[start:]:
        if column >= len(row):
            continue
        ticker = clean_ticker(str(row[column] or ""))
        if ticker in _NOT_HOLDINGS or not _TICKER.match(ticker) or ticker in found:
            continue
        found.append(ticker)
    return tuple(found)


@lru_cache(maxsize=4)
def _cached_tickers(path: str, _mtime: int) -> tuple[str, ...]:
    tickers = tickers_from_rows(_read_rows(Path(path)))
    if not tickers:
        raise PortfolioError(f"No stock symbols were found in {path}.")
    return tickers


def _read_rows(path: Path) -> list[list]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if any(name.startswith("Index/") for name in names):
            return _numbers_rows(path)
        raise PortfolioError("Excel workbooks are not supported yet. Save the portfolio as CSV or Numbers.")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return [row for row in csv.reader(io.StringIO(text))]


def _numbers_rows(path: Path) -> list[list]:
    from numbers_parser import Document

    # numbers-parser only opens files whose name ends in .numbers.
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "portfolio.numbers"
        shutil.copyfile(path, copy)
        document = Document(str(copy))
        rows: list[list] = []
        for sheet in document.sheets:
            for table in sheet.tables:
                rows.extend(list(row) for row in table.rows(values_only=True))
        return rows
