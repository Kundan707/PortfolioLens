"""Public headlines for a contract counterparty from the last year."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
_USER_AGENT = "Mag7EarningsAgent research@example.com"
_GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_TAG = re.compile(r"<[^>]+>")
_DEAL = re.compile(
    r"\b(contracts?|agreements?|deals?|partnerships?|partners?|customers?|"
    r"suppliers?|acquir\w+|acquisition|invest\w+|signed|signs|signing)\b",
    re.IGNORECASE,
)
_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "will", "their",
    "about", "into", "company", "companies", "filing", "filed", "source",
    "which", "after", "before", "during", "year", "years", "last", "public",
}


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    publisher: str
    published: str
    summary: str
    url: str = ""


def fetch_counterparty_news(company: str, *, as_of: date | None = None, limit: int = 6) -> list[NewsItem]:
    """Return public headlines about one company from the past year."""
    name = " ".join((company or "").split())
    if not name:
        return []
    day = as_of or datetime.now(timezone.utc).date()
    try:
        items = _from_google(name, day, limit)
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError):
        items = []
    if items:
        return items
    try:
        return _from_gdelt(name, day, limit)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return []


def select_relevant(company: str, items: list[NewsItem], context: str) -> list[NewsItem]:
    """Keep headlines that match the filing or describe a deal, and that have a link."""
    company_words = {word.lower() for word in re.findall(r"[A-Za-z0-9]{3,}", company)}
    context_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9&]{3,}", context)
        if word.lower() not in _STOP and word.lower() not in company_words
    }
    chosen: list[NewsItem] = []
    for item in items:
        if not item.url.startswith(("https://", "http://")):
            continue
        blob = f"{item.title} {item.summary}"
        words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9&]{3,}", blob)}
        if _DEAL.search(blob) or len(words & context_words) >= 2:
            chosen.append(item)
    return chosen


def format_news(company: str, items: list[NewsItem]) -> str:
    if not items:
        return (
            f"Source: Public news for {company} | Page: 1\n"
            "No relevant public headlines from the last year were found."
        )
    blocks = []
    for item in items:
        summary = item.summary or item.title
        link = f"Link: {item.url}\n" if item.url.startswith(("https://", "http://")) else ""
        blocks.append(
            f"Source: {item.publisher} — {company} — {item.published} | Page: 1\n"
            f"{link}{item.title}\n{summary}"
        )
    return "\n---\n".join(blocks)


def headlines_from_rss(xml_text: str, *, as_of: date, limit: int = 6) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    cutoff = as_of - timedelta(days=365)
    items: list[NewsItem] = []
    for node in root.findall("./channel/item"):
        title = _clean(node.findtext("title") or "")
        if not title:
            continue
        published = _rss_date(node.findtext("pubDate") or "")
        if published is not None and published < cutoff:
            continue
        publisher = _clean(node.findtext("source") or "")
        if not publisher and " - " in title:
            title, publisher = title.rsplit(" - ", 1)
            title = title.strip()
            publisher = publisher.strip()
        summary = _clean(node.findtext("description") or "")
        if summary == title:
            summary = ""
        link = (node.findtext("link") or node.findtext("guid") or "").strip()
        items.append(
            NewsItem(
                title=title,
                publisher=publisher or "Public news",
                published=published.isoformat() if published else "last year",
                summary=summary[:300],
                url=link if link.startswith(("https://", "http://")) else "",
            )
        )
        if len(items) >= limit:
            break
    return items


def _from_google(company: str, day: date, limit: int) -> list[NewsItem]:
    query = urllib.parse.quote(f'"{company}" when:1y')
    payload = _get(_GOOGLE_NEWS.format(query=query), accept="application/rss+xml")
    return headlines_from_rss(payload, as_of=day, limit=limit)


def _from_gdelt(company: str, day: date, limit: int) -> list[NewsItem]:
    start = datetime.combine(day - timedelta(days=365), datetime.min.time(), timezone.utc)
    params = urllib.parse.urlencode(
        {
            "query": f'"{company}"',
            "mode": "ArtList",
            "maxrecords": str(limit),
            "format": "json",
            "sort": "datedesc",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        }
    )
    payload = json.loads(_get(f"{_GDELT}?{params}", accept="application/json"))
    items: list[NewsItem] = []
    for article in payload.get("articles") or []:
        title = _clean(str(article.get("title") or ""))
        if not title:
            continue
        seen = str(article.get("seendate") or "")
        published = seen[:8]
        if len(published) == 8 and published.isdigit():
            published = f"{published[:4]}-{published[4:6]}-{published[6:8]}"
        else:
            published = "last year"
        link = str(article.get("url") or "").strip()
        items.append(
            NewsItem(
                title=title,
                publisher=_clean(str(article.get("domain") or "Public news")),
                published=published,
                summary="",
                url=link if link.startswith(("https://", "http://")) else "",
            )
        )
    return items


def _rss_date(value: str) -> date | None:
    if not value.strip():
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.date()


def _clean(value: str) -> str:
    text = _TAG.sub(" ", value)
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&amp;", "&")
    return " ".join(text.split())


def _get(url: str, *, accept: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")
