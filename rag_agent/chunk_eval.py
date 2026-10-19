"""Compare fixed-size and semantic chunking on the portfolio filings.

Builds one Chroma index per strategy from the same SEC data, then sends the same questions
through the agent's search tool against each index and scores whether the expected
company or fact comes back. With an Anthropic key and --llm, Claude also answers each question.

    python -m rag_agent.chunk_eval [--llm]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import date
from pathlib import Path

from rag_agent.config import ROOT, api_key
from rag_agent.earnings import (
    _cutoff,
    _filing_urls,
    company_for,
    fetch_company_facts,
    fetch_submissions,
    filing_documents,
    statement_documents,
)
from rag_agent.portfolio import portfolio_tickers
from rag_agent.store import clear_library, get_embeddings, get_vectorstore, replace_ticker, reset_client

OUT = ROOT / "data" / "eval"
STRATEGIES = ("fixed", "semantic")

# Descriptive questions do not name the counterparty, so the chunk has to carry the meaning.
QUESTIONS: list[dict] = [
    {"ticker": "AAPL", "focus": "contracts", "kind": "descriptive", "expect": ["Google"],
     "query": "Which company pays Apple to be the default search engine in Safari?"},
    {"ticker": "MSFT", "focus": "contracts", "kind": "descriptive", "expect": ["OpenAI"],
     "query": "Which AI research lab does Microsoft have a strategic partnership and investment in?"},
    {"ticker": "NVDA", "focus": "contracts", "kind": "descriptive", "expect": ["TSMC", "Taiwan Semiconductor"],
     "query": "Which foundry manufactures NVIDIA's wafers?"},
    {"ticker": "NVDA", "focus": "contracts", "kind": "descriptive", "expect": ["SK Hynix", "Samsung", "Micron"],
     "query": "Which memory suppliers provide NVIDIA with high-bandwidth memory?"},
    {"ticker": "NVDA", "focus": "contracts", "kind": "descriptive", "expect": ["OpenAI"],
     "query": "Which AI company did NVIDIA agree to invest in to deploy data center systems?"},
    {"ticker": "NVDA", "focus": "contracts", "kind": "descriptive", "expect": ["Hon Hai", "Wistron", "Foxconn"],
     "query": "Which contract manufacturers assemble NVIDIA's systems and boards?"},
    {"ticker": "TSLA", "focus": "contracts", "kind": "descriptive", "expect": ["Panasonic"],
     "query": "Who supplies Tesla with battery cells under a long-term supply agreement?"},
    {"ticker": "TSLA", "focus": "contracts", "kind": "descriptive", "expect": ["xAI", "X.AI"],
     "query": "Which artificial intelligence company related to Elon Musk did Tesla transact with?"},
    {"ticker": "SPCX", "focus": "contracts", "kind": "descriptive", "expect": ["EchoStar"],
     "query": "Whose wireless spectrum licenses did SpaceX agree to acquire?"},
    {"ticker": "SPCX", "focus": "contracts", "kind": "descriptive", "expect": ["xAI", "X.AI"],
     "query": "Which AI company did SpaceX merge or combine with?"},
    {"ticker": "AAPL", "focus": "contracts", "kind": "named", "expect": ["Google"],
     "query": "Apple agreement with Google"},
    {"ticker": "MSFT", "focus": "contracts", "kind": "named", "expect": ["OpenAI"],
     "query": "Microsoft OpenAI agreement"},
    {"ticker": "NVDA", "focus": "contracts", "kind": "named", "expect": ["Intel"],
     "query": "NVIDIA agreement with Intel"},
    {"ticker": "TSLA", "focus": "contracts", "kind": "named", "expect": ["Toyota"],
     "query": "Tesla agreement with Toyota"},
    {"ticker": "SPCX", "focus": "contracts", "kind": "named", "expect": ["Mesh Optical"],
     "query": "SpaceX acquisition of Mesh Optical Technologies"},
    {"ticker": "AAPL", "focus": "latest_year", "kind": "earnings", "expect": ["Revenue:"],
     "query": "Apple revenue in the latest fiscal year"},
    {"ticker": "NVDA", "focus": "latest_quarter", "kind": "earnings", "expect": ["Diluted EPS"],
     "query": "NVIDIA diluted EPS latest quarter"},
    {"ticker": "SPCX", "focus": "latest_quarter", "kind": "earnings", "expect": ["Revenue streams"],
     "query": "SpaceX revenue streams latest quarter"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="Also have Claude answer every question")
    parser.add_argument("--skip-build", action="store_true", help="Reuse indexes from a previous run")
    args = parser.parse_args(argv)

    tickers = portfolio_tickers()
    results: dict = {"tickers": list(tickers), "strategies": {}}
    print("Warming up the embedding model…", flush=True)
    get_embeddings().embed_documents(["warm up"])
    sources = None if args.skip_build else _fetch_sources(tickers)

    for strategy in STRATEGIES:
        _use(strategy)
        entry: dict = {}
        if sources is not None:
            print(f"\nBuilding the {strategy} index…", flush=True)
            entry["build"] = _build(sources)
        entry["index"] = _index_stats(OUT / f"chroma_{strategy}")
        print(f"Asking {len(QUESTIONS)} questions against the {strategy} index…", flush=True)
        entry["retrieval"] = _retrieval()
        if args.llm:
            entry["answers"] = _answers()
        results["strategies"][strategy] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "chunking_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_summary(results)
    print(f"\nFull results: {path}")
    if args.llm and not api_key().startswith("sk-ant-"):
        print("No valid Anthropic key, so Claude answers were skipped.")
    return 0


def _use(strategy: str) -> None:
    os.environ["CHUNKING"] = strategy
    os.environ["CHROMA_DIR"] = str(OUT / f"chroma_{strategy}")
    reset_client()


def _fetch_sources(tickers) -> list[dict]:
    cutoff = _cutoff(date.today())
    sources = []
    for ticker in tickers:
        found = company_for(ticker)
        if found is None:
            continue
        cik, company = found
        print(f"Fetching SEC data for {ticker}…", flush=True)
        sources.append(
            {
                "ticker": ticker,
                "company": company,
                "cik": cik,
                "facts": fetch_company_facts(cik),
                "submissions": fetch_submissions(cik),
                "cutoff": cutoff,
            }
        )
    return sources


def _build(sources: list[dict]) -> dict:
    clear_library()
    chunk_seconds = 0.0
    store_seconds = 0.0
    counts: dict[str, int] = {}
    for item in sources:
        started = time.perf_counter()
        passages, streams = filing_documents(
            item["ticker"], item["company"], item["cik"], item["submissions"], item["cutoff"], pause_seconds=0
        )
        statements = statement_documents(
            item["ticker"],
            item["company"],
            item["facts"],
            streams=streams,
            filing_urls=_filing_urls(item["cik"], item["submissions"]),
        )
        chunked = time.perf_counter()
        built = statements + passages
        replace_ticker(item["ticker"], built, [f"{item['ticker']}:{index}" for index in range(len(built))])
        stored = time.perf_counter()
        chunk_seconds += chunked - started
        store_seconds += stored - chunked
        counts[item["ticker"]] = len(built)
        print(f"  {item['ticker']}: {len(built)} chunks", flush=True)
    return {
        "chunking_seconds": round(chunk_seconds, 1),
        "embed_and_store_seconds": round(store_seconds, 1),
        "chunks_per_ticker": counts,
    }


def _index_stats(folder: Path) -> dict:
    found = get_vectorstore().get(include=["documents", "metadatas"])
    contract = [
        text
        for text, meta in zip(found["documents"], found["metadatas"])
        if (meta or {}).get("focus_tag") == "contracts"
    ]
    statements = len(found["documents"]) - len(contract)
    sizes = sorted(len(text) for text in contract)
    bodies = [text.split("\n\n", 1)[-1].rstrip() for text in contract]
    return {
        "total_chunks": len(found["documents"]),
        "statement_chunks": statements,
        "contract_chunks": len(contract),
        "chars_mean": round(statistics.mean(sizes)) if sizes else 0,
        "chars_median": round(statistics.median(sizes)) if sizes else 0,
        "chars_p10": _percentile(sizes, 10),
        "chars_p90": _percentile(sizes, 90),
        "chars_min": sizes[0] if sizes else 0,
        "chars_max": sizes[-1] if sizes else 0,
        "header_only_chunks": sum(1 for text in contract if text.startswith("Ticker:") and "\n\n" not in text.strip()),
        "header_only_chunks": sum(1 for text in contract if text.startswith("Ticker:") and "\n\n" not in text.strip()),
        "with_header_pct": _pct(sum(text.startswith("Ticker:") for text in contract), len(contract)),
        "ends_mid_sentence_pct": _pct(sum(not body.endswith((".", "!", "?", '"', "”", ")")) for body in bodies), len(bodies)),
        "disk_mb": round(sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6, 1),
    }


def _retrieval() -> dict:
    from rag_agent.graph import search_earnings

    rows = []
    for question in QUESTIONS:
        started = time.perf_counter()
        text = search_earnings.invoke(
            {"query": question["query"], "ticker": question["ticker"], "focus": question["focus"]}
        )
        latency = (time.perf_counter() - started) * 1000
        blocks = [block for block in text.split("\n---\n") if block.startswith("Source:")]
        terms = [term.lower() for term in question["expect"]]
        hits = [any(term in block.lower() for term in terms) for block in blocks]
        rank = hits.index(True) + 1 if True in hits else None
        rows.append(
            {
                "ticker": question["ticker"],
                "kind": question["kind"],
                "query": question["query"],
                "returned": len(blocks),
                "first_hit_rank": rank,
                "relevant_chunks": sum(hits),
                "context_chars": sum(len(block) for block in blocks),
                "latency_ms": round(latency, 1),
            }
        )
    return {"summary": _summarize(rows), "by_kind": _by_kind(rows), "questions": rows}


def _answers() -> list[dict]:
    if not api_key().startswith("sk-ant-"):
        return []
    from rag_agent.graph import answer

    rows = []
    for question in QUESTIONS:
        started = time.perf_counter()
        try:
            result = answer(f"Ticker: {question['ticker']}. {question['query']}")
            text = result.answer
        except Exception as exc:
            text = f"ERROR: {exc}"
        rows.append(
            {
                "query": question["query"],
                "mentions_expected": any(term.lower() in text.lower() for term in question["expect"]),
                "seconds": round(time.perf_counter() - started, 1),
                "answer": text[:600],
            }
        )
    return rows


def _summarize(rows: list[dict]) -> dict:
    count = len(rows) or 1
    ranks = [row["first_hit_rank"] for row in rows]
    return {
        "questions": len(rows),
        "hit_at_1_pct": _pct(sum(1 for r in ranks if r == 1), count),
        "hit_at_3_pct": _pct(sum(1 for r in ranks if r and r <= 3), count),
        "hit_at_k_pct": _pct(sum(1 for r in ranks if r), count),
        "mrr": round(sum(1 / r for r in ranks if r) / count, 3),
        "context_precision_pct": _pct(
            sum(row["relevant_chunks"] for row in rows), sum(row["returned"] for row in rows) or 1
        ),
        "avg_context_chars": round(statistics.mean(row["context_chars"] for row in rows)),
        "latency_ms_median": round(statistics.median(row["latency_ms"] for row in rows), 1),
    }


def _by_kind(rows: list[dict]) -> dict:
    kinds: dict[str, list[dict]] = {}
    for row in rows:
        kinds.setdefault(row["kind"], []).append(row)
    return {kind: _summarize(items) for kind, items in kinds.items()}


def _print_summary(results: dict) -> None:
    print("\nSummary")
    for strategy, entry in results["strategies"].items():
        print(f"\n{strategy}")
        for section in ("build", "index"):
            if section in entry:
                print(f"  {section}: {entry[section]}")
        print(f"  retrieval: {entry['retrieval']['summary']}")
        for kind, summary in entry["retrieval"]["by_kind"].items():
            print(f"    {kind}: {summary}")


def _percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
    return values[min(len(values) - 1, int(len(values) * pct / 100))]


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
