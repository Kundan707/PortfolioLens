"""Format and parse the passages returned by the PDF search tool."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    source: str
    page: int | None
    excerpt: str
    url: str = ""


def format_passages(docs: list) -> str:
    blocks: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        text = (getattr(doc, "page_content", "") or "").strip()
        url = str(metadata.get("url") or "").strip()
        link = f"Link: {url}\n" if url.startswith(("https://", "http://")) else ""
        blocks.append(f"Source: {source} | Page: {page}\n{link}{text}")
    return "\n---\n".join(blocks)


def parse_sources(content: str) -> list[Source]:
    if not content or "earnings library is empty" in content or "PDF library is empty" in content:
        return []
    sources: list[Source] = []
    for block in content.split("\n---\n"):
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("Source:"):
            continue
        name, page = _header(lines[0])
        url, body = _link(lines[1:])
        excerpt = "\n".join(body).strip()
        if not excerpt:
            continue
        sources.append(Source(source=name, page=page, excerpt=excerpt[:700], url=url))
    return sources


def _link(lines: list[str]) -> tuple[str, list[str]]:
    url = ""
    body: list[str] = []
    for line in lines:
        if line.startswith("Link: ") and not url:
            candidate = line.removeprefix("Link: ").strip()
            if candidate.startswith(("https://", "http://")):
                url = candidate
                continue
        body.append(line)
    return url, body


def _header(line: str) -> tuple[str, int | None]:
    if "|" not in line:
        return line.strip().removeprefix("Source:").strip(), None
    left, right = line.split("|", 1)
    name = left.strip().removeprefix("Source:").strip()
    page_text = right.strip().removeprefix("Page:").strip()
    try:
        return name, int(page_text)
    except ValueError:
        return name, None
