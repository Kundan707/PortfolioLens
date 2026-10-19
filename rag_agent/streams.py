"""Revenue streams from inline XBRL in 10-K and 10-Q filings."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CONTEXT = re.compile(
    r'<xbrli:context\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</xbrli:context>',
    re.IGNORECASE | re.DOTALL,
)
_FACT = re.compile(
    r"<ix:nonFraction\b([^>]*)>([^<]*)</ix:nonFraction>",
    re.IGNORECASE | re.DOTALL,
)
_ATTR = re.compile(r'([A-Za-z]+)="([^"]*)"')
_MEMBER = re.compile(
    r'dimension="([^"]+)"[^>]*>([^<]+)',
    re.IGNORECASE,
)
_REVENUE_NAMES = {
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
}
_AXES = {"ProductOrServiceAxis", "StatementBusinessSegmentsAxis"}
_CAMEL_A = re.compile(r"(?<=[a-z])(?=[A-Z])")
_CAMEL_B = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True, slots=True)
class RevenueStream:
    end: str
    kind: str
    label: str
    value: float


def revenue_streams(html: str) -> list[RevenueStream]:
    """Return product or segment revenue lines for quarterly and annual periods."""
    contexts = _contexts(html)
    totals: dict[tuple[str, str], float] = {}
    buckets: dict[tuple[str, str, str], dict[str, float]] = {}
    for attrs, raw_value in _facts(html):
        name = attrs.get("name") or ""
        if name not in _REVENUE_NAMES:
            continue
        context = contexts.get(attrs.get("contextRef") or "")
        if context is None:
            continue
        value = _number(raw_value, attrs.get("scale"), attrs.get("sign"))
        if value is None:
            continue
        key = (context["end"], context["kind"])
        if not context["members"]:
            current = totals.get(key)
            if current is None or abs(value) > abs(current):
                totals[key] = value
            continue
        included = [item for item in context["members"] if item[0] in _AXES]
        product = [member for axis, member in included if axis == "ProductOrServiceAxis"]
        if len(product) == 1:
            axis, member = "ProductOrServiceAxis", product[0]
        elif len(included) == 1:
            axis, member = included[0]
        else:
            continue
        label = _pretty(member)
        group = buckets.setdefault((context["end"], context["kind"], axis), {})
        current = group.get(label)
        if current is None or abs(value) > abs(current):
            group[label] = value

    chosen: list[RevenueStream] = []
    periods = {(end, kind) for end, kind, _axis in buckets}
    for end, kind in periods:
        axes = {
            axis: list(items.items())
            for (item_end, item_kind, axis), items in buckets.items()
            if item_end == end and item_kind == kind
        }
        for label, value in _choose(axes, totals.get((end, kind))):
            chosen.append(RevenueStream(end, kind, label, value))
    chosen.sort(key=lambda item: (item.end, item.kind, -item.value))
    return chosen


def _contexts(html: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for match in _CONTEXT.finditer(html):
        body = match.group(2)
        start = _tag_value(body, "startDate")
        end = _tag_value(body, "endDate")
        if not start or not end:
            continue
        kind = _kind(start, end)
        if kind is None:
            continue
        members = []
        for axis, member in _MEMBER.findall(body):
            members.append((axis.split(":")[-1], member.strip()))
        found[match.group(1)] = {"end": end, "kind": kind, "members": members}
    return found


def _facts(html: str) -> list[tuple[dict[str, str], str]]:
    facts = []
    for match in _FACT.finditer(html):
        attrs = dict(_ATTR.findall(match.group(1)))
        facts.append((attrs, match.group(2)))
    return facts


def _choose(axes: dict[str, list[tuple[str, float]]], total: float | None) -> list[tuple[str, float]]:
    scored: list[tuple[str, list[tuple[str, float]], float]] = []
    for axis, items in axes.items():
        rollups = {"Product", "Services", "Service Other"}
        without_rollups = [item for item in items if item[0] not in rollups]
        with_services = [item for item in items if item[0] not in {"Product", "Service Other"}]
        source = items
        if total is not None:
            for candidate in (without_rollups, with_services):
                covered = sum(value for _, value in candidate)
                if len(candidate) >= 2 and abs(covered - total) / abs(total) <= 0.05:
                    source = candidate
                    break
        kept: list[tuple[str, float]] = []
        for label, value in source:
            others = [other for other_label, other in source if other_label != label]
            if total and abs(value - total) / abs(total) <= 0.002:
                continue
            if _is_subtotal(value, others):
                continue
            kept.append((label, value))
        if kept:
            scored.append((axis, kept, sum(value for _, value in kept)))
    if not scored:
        return []
    completed: list[tuple[str, list[tuple[str, float]], float]] = []
    for axis, kept, covered in scored:
        combo = list(kept)
        combo_sum = covered
        if total and combo_sum < total * 0.98:
            for other_axis, other_kept, _other_sum in scored:
                if other_axis == axis:
                    continue
                for label, value in other_kept:
                    if _overlaps(label, value, combo):
                        continue
                    if combo_sum + value <= total * 1.03:
                        combo.append((label, value))
                        combo_sum += value
        completed.append((axis, combo, combo_sum))
    completed.sort(key=lambda item: _rank(item, total))
    _axis, kept, _covered = completed[0]
    kept.sort(key=lambda item: item[1], reverse=True)
    return kept


def _overlaps(label: str, value: float, kept: list[tuple[str, float]]) -> bool:
    words = {word.lower() for word in label.split() if len(word) > 3}
    for existing, existing_value in kept:
        if abs(existing_value - value) / max(value, 1) <= 0.03:
            return True
        existing_words = {word.lower() for word in existing.split() if len(word) > 3}
        if words and words <= existing_words:
            return True
    return False


def _rank(item: tuple[str, list[tuple[str, float]], float], total: float | None) -> tuple:
    _axis, kept, covered = item
    if not total:
        return (0, -len(kept))
    gap = abs(covered - total) / abs(total)
    product_first = 0 if _axis == "ProductOrServiceAxis" else 1
    return (gap > 0.12, round(gap, 3), product_first, -len(kept))


def _is_subtotal(value: float, others: list[float]) -> bool:
    if value <= 0:
        return False
    target = int(round(value / 1_000_000))
    pieces = [
        int(round(other / 1_000_000))
        for other in others
        if 0 < other < value * 0.98
    ]
    if len(pieces) < 2:
        return False
    reachable = {0}
    for piece in pieces:
        reachable.update({got + piece for got in reachable if got + piece <= target * 1.02})
    return any(abs(got - target) <= max(1, target * 0.004) and got for got in reachable)


def _kind(start: str, end: str) -> str | None:
    try:
        days = (_date(end) - _date(start)).days
    except ValueError:
        return None
    if 80 <= days <= 120:
        return "quarter"
    if 340 <= days <= 380:
        return "annual"
    return None


def _date(value: str):
    from datetime import date

    year, month, day = (int(part) for part in value.split("-"))
    return date(year, month, day)


def _tag_value(body: str, tag: str) -> str:
    match = re.search(rf"<xbrli:{tag}>([^<]+)", body, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _number(text: str, scale: str | None, sign: str | None) -> float | None:
    cleaned = text.replace(",", "").replace(" ", "").strip()
    if not cleaned or cleaned in {"-", "—"}:
        return None
    negative = cleaned.startswith("(") or sign == "-"
    cleaned = cleaned.strip("()")
    try:
        value = float(cleaned) * (10 ** int(scale or "0"))
    except ValueError:
        return None
    return -value if negative else value


def _pretty(member: str) -> str:
    name = member.split(":")[-1]
    name = re.sub(r"(Member|Segment)$", "", name)
    name = re.sub(r"and(?=[A-Z])", " and ", name)
    name = _CAMEL_B.sub(" ", name)
    name = _CAMEL_A.sub(" ", name)
    name = name.replace("Three Six Five", "365")
    name = name.replace("You Tube", "YouTube")
    name = name.replace("Linked In", "LinkedIn")
    name = re.sub(r" Revenue$", "", name)
    name = re.sub(r" Segment$", "", name)
    fixes = {
        "I Phone": "iPhone",
        "I Pad": "iPad",
        "You Tube": "YouTube",
        "XBOX": "Xbox",
        "OEM And Other": "OEM and other",
        "Service": "Services",
    }
    return fixes.get(name, name)
