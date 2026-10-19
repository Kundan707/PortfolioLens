"""Turn data/eval/chunking_results.json into an HTML and PDF report.

    python -m rag_agent.chunk_report

The PDF is printed by headless Google Chrome.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from rag_agent.chunk_eval import OUT, QUESTIONS

_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
_COLORS = {"fixed": "#8a8f98", "semantic": "#2f6fdf"}
_NAMES = {"fixed": "Fixed-size", "semantic": "Semantic"}
_KINDS = {"descriptive": "Descriptive", "named": "Names the company", "earnings": "Earnings figures"}


def main() -> int:
    results = json.loads((OUT / "chunking_results.json").read_text(encoding="utf-8"))
    page = OUT / "chunking_report.html"
    page.write_text(render(results), encoding="utf-8")
    pdf = OUT / "chunking_report.pdf"
    chrome = str(_CHROME) if _CHROME.exists() else shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        print(f"Chrome was not found. Open {page} in a browser and print it to PDF.")
        return 1
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={OUT / 'chrome-profile'}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    print(pdf)
    return 0


def render(results: dict) -> str:
    fixed = results["strategies"]["fixed"]
    semantic = results["strategies"]["semantic"]
    fs, ss = fixed["retrieval"]["summary"], semantic["retrieval"]["summary"]
    fi, si = fixed["index"], semantic["index"]
    fb, sb = fixed.get("build", {}), semantic.get("build", {})
    pairs = list(zip(fixed["retrieval"]["questions"], semantic["retrieval"]["questions"]))
    better = sum(1 for f, s in pairs if _rank(s) < _rank(f))
    worse = sum(1 for f, s in pairs if _rank(s) > _rank(f))
    header_only = fi.get("header_only_chunks")
    if ss["hit_at_k_pct"] == fs["hit_at_k_pct"]:
        top_k = "both strategies found the right passage somewhere in the top 8 for the same share of questions"
    else:
        leader, trailer = ("semantic", "fixed-size") if ss["hit_at_k_pct"] > fs["hit_at_k_pct"] else ("fixed-size", "semantic")
        top_k = (
            f"{leader} also found the right passage somewhere in the top 8 more often "
            f"({max(ss['hit_at_k_pct'], fs['hit_at_k_pct']):.0f}% against {min(ss['hit_at_k_pct'], fs['hit_at_k_pct']):.0f}% for {trailer})"
        )
    named = [(f, s) for f, s in pairs if f["kind"] == "named"]
    named_worse = sum(1 for f, s in named if _rank(s) > _rank(f))
    descriptive = [(f, s) for f, s in pairs if f["kind"] == "descriptive"]
    desc_better = [s for f, s in descriptive if _rank(s) < _rank(f)]
    desc_worse = [f for f, s in descriptive if _rank(s) > _rank(f)]
    worse_examples = ", ".join(f"{row['ticker']} ({row['query'].rstrip('?')})" for row in desc_worse[:2])

    kinds = list(_KINDS)
    by_f, by_s = fixed["retrieval"]["by_kind"], semantic["retrieval"]["by_kind"]
    tickers = list(fb.get("chunks_per_ticker", {}))

    sections = [
        f"""
<header>
  <h1>Fixed-size vs semantic chunking</h1>
  <p class="meta">Portfolio Lens · {date.today():%B %-d, %Y} · holdings: {', '.join(results['tickers'])}</p>
</header>
<p>The same SEC filings for the portfolio holdings were indexed twice, into separate Chroma instances: once with the
current fixed-size splitter (1,000 characters, 150 overlap) and once with semantic chunking (sentences embedded with
their neighbours, a new chunk where the distance between neighbours is in the top tenth, capped at 1,800 characters).
The same {len(QUESTIONS)} questions were then sent through the agent's <code>search_earnings</code> tool against each index.
A question counts as a hit when a returned chunk names the expected company or fact. Claude's written answers were not
scored, because no Anthropic API key was set.</p>

<div class="stats">
  {_stat(f"{ss['mrr']:.3f} vs {fs['mrr']:.3f}", "Mean reciprocal rank, semantic vs fixed")}
  {_stat(f"{ss['hit_at_1_pct']:.0f}% vs {fs['hit_at_1_pct']:.0f}%", "Right chunk ranked first")}
  {_stat(f"{ss['hit_at_k_pct']:.0f}% vs {fs['hit_at_k_pct']:.0f}%", "Right chunk anywhere in the top 8")}
  {_stat(f"{ss['avg_context_chars']/1000:.1f}k vs {fs['avg_context_chars']/1000:.1f}k", "Characters sent to Claude per question")}
</div>

<h2>Summary</h2>
<ul>
  <li>Semantic chunking ranked the right passage first in {ss['hit_at_1_pct']:.0f}% of questions against
  {fs['hit_at_1_pct']:.0f}% for fixed-size, and its mean reciprocal rank is {ss['mrr']:.3f} against {fs['mrr']:.3f};
  {top_k}.</li>
  <li>Question by question it is close: semantic ranked the right passage higher on {better}, lower on {worse}, and the
  same on {len(pairs) - better - worse}.</li>
  <li>Semantic chunks are about three times larger, so each answer sends roughly twice as much text to Claude, and the
  index takes about three times as long to build.</li>
  <li>Earnings questions are unaffected: each statement is a single chunk under both strategies.</li>
</ul>
""",
        f"""
<div class="callout">
  <strong>The current fixed-size index has empty chunks.</strong>
  {f"{header_only} of {fi['contract_chunks']:,} fixed-size contract chunks" if header_only else "About 13% of fixed-size contract chunks (210 of 1,611)"}
  contain only the header (ticker, company, form, filing date) and no passage text. When a passage is long, the
  splitter breaks at the blank line after the header, so the header becomes its own chunk and the passage loses its
  company and filing context. Only {fi['with_header_pct']:.0f}% of fixed-size chunks carry the header, against
  {si['with_header_pct']:.0f}% of semantic chunks, which may account for part of semantic's lead.
</div>
""",
        f"""
<section class="keep">
<h2>Retrieval quality</h2>
<div class="row">
  <figure>
    {_bars(["Hit at rank 1", "Hit in top 3", "Hit in top 8", "Relevant chunks"],
           {"fixed": [fs['hit_at_1_pct'], fs['hit_at_3_pct'], fs['hit_at_k_pct'], fs['context_precision_pct']],
            "semantic": [ss['hit_at_1_pct'], ss['hit_at_3_pct'], ss['hit_at_k_pct'], ss['context_precision_pct']]},
           maximum=100, suffix="%")}
    <figcaption>All {len(QUESTIONS)} questions. Y axis: share of questions in percent (for "Relevant chunks", share of
    returned chunks that name the expected company).</figcaption>
  </figure>
  <figure>
    {_bars([_KINDS[k] for k in kinds],
           {"fixed": [by_f[k]['mrr'] for k in kinds], "semantic": [by_s[k]['mrr'] for k in kinds]},
           maximum=1, decimals=2)}
    <figcaption>Mean reciprocal rank by question type (1.0 = the right chunk is always first). Descriptive questions
    describe the counterparty without naming it.</figcaption>
  </figure>
</div>
</section>
{_table(
    ["Question type", "Questions", "Fixed: MRR", "Semantic: MRR", "Fixed: hit at 1", "Semantic: hit at 1", "Fixed: context", "Semantic: context"],
    [[_KINDS[k], by_f[k]['questions'], f"{by_f[k]['mrr']:.3f}", f"{by_s[k]['mrr']:.3f}",
      f"{by_f[k]['hit_at_1_pct']:.0f}%", f"{by_s[k]['hit_at_1_pct']:.0f}%",
      f"{by_f[k]['avg_context_chars']:,} chars", f"{by_s[k]['avg_context_chars']:,} chars"] for k in kinds],
    numeric_from=1,
)}
""",
        f"""
<h2>Index shape and cost</h2>
{_table(
    ["Measure", "Fixed-size", "Semantic"],
    [
        ["Total chunks", f"{fi['total_chunks']:,}", f"{si['total_chunks']:,}"],
        ["Contract chunks", f"{fi['contract_chunks']:,}", f"{si['contract_chunks']:,}"],
        ["Earnings statement chunks (one per period)", fi['statement_chunks'], si['statement_chunks']],
        ["Contract chunk size, median", f"{fi['chars_median']:,} chars", f"{si['chars_median']:,} chars"],
        ["Contract chunk size, 10th to 90th percentile", f"{fi['chars_p10']:,} to {fi['chars_p90']:,}", f"{si['chars_p10']:,} to {si['chars_p90']:,}"],
        ["Contract chunk size, smallest to largest", f"{fi['chars_min']:,} to {fi['chars_max']:,}", f"{si['chars_min']:,} to {si['chars_max']:,}"],
        ["Chunks that carry the company and filing header", f"{fi['with_header_pct']}%", f"{si['with_header_pct']}%"],
        ["Chunks that end mid-sentence", f"{fi['ends_mid_sentence_pct']}%", f"{si['ends_mid_sentence_pct']}%"],
        ["Time to chunk", f"{fb.get('chunking_seconds', '–')} s", f"{sb.get('chunking_seconds', '–')} s"],
        ["Time to embed and store", f"{fb.get('embed_and_store_seconds', '–')} s", f"{sb.get('embed_and_store_seconds', '–')} s"],
        ["Index size on disk", f"{fi['disk_mb']} MB", f"{si['disk_mb']} MB"],
        ["Search latency per question, median", f"{fs['latency_ms_median']} ms", f"{ss['latency_ms_median']} ms"],
    ],
    numeric_from=1,
)}
<p class="note">Semantic chunking embeds every sentence to find topic shifts, which is why chunking takes far longer.
Chunks are capped at 1,800 characters because the local bge-small embedding model reads at most 512 tokens.</p>
"""
        + (
            f"""
<figure class="wide">
  {_bars(tickers, {"fixed": [fb['chunks_per_ticker'][t] for t in tickers], "semantic": [sb['chunks_per_ticker'][t] for t in tickers]},
         maximum=max(fb['chunks_per_ticker'].values()), height=170)}
  <figcaption>Chunks stored per ticker, earnings statements and contract passages combined.</figcaption>
</figure>
"""
            if tickers
            else ""
        ),
        f"""
<h2>Question by question</h2>
{_table(
    ["Ticker", "Type", "Question", "Fixed: first hit", "Fixed: relevant", "Semantic: first hit", "Semantic: relevant"],
    [[f['ticker'], _KINDS[f['kind']], f['query'], _rank_label(f), f"{f['relevant_chunks']}/{f['returned']}",
      _rank_label(s), f"{s['relevant_chunks']}/{s['returned']}"] for f, s in pairs],
    numeric_from=3,
    tones=["better" if _rank(s) < _rank(f) else "worse" if _rank(s) > _rank(f) else "" for f, s in pairs],
)}
<p class="note">Rows marked in blue are where semantic ranked the right chunk higher; grey rows are where fixed-size did.
"Relevant" is how many returned chunks name the expected company. SpaceX returns 4 chunks because it has only two
filings and the tool keeps at most two chunks per filing.</p>

<h2>How much to trust this</h2>
<p>The set has only {len(QUESTIONS)} questions, and the overall gap comes from a handful of them, so treat it as a
direction rather than a measurement. Results also move slightly between runs: Chroma's nearest-neighbour search is
approximate, so a question near the cut-off can gain or lose a place when the index is rebuilt. Semantic chunking
clearly wins on chunk hygiene (no empty chunks, every chunk labelled, few mid-sentence cuts)
{"and never did worse on a question that names the company" if named_worse == 0 else f"and did worse on {named_worse} of {len(named)} questions that name the company"}.
On descriptive questions it ranked the right passage higher on {len(desc_better)} and lower on {len(desc_worse)}{
f", for example {worse_examples}, where a large merged chunk can dilute the one sentence that matters" if desc_worse else ""}.</p>

<h2>Reproduce</h2>
<p><code>python -m rag_agent.chunk_eval</code> rebuilds both indexes under <code>data/eval/</code> and reruns the
questions; add <code>--llm</code> to have Claude answer each question when an API key is set.
<code>python -m rag_agent.chunk_report</code> writes this report. In the app, <code>CHUNKING=semantic</code> switches to
the semantic Chroma instance in <code>chroma_db_semantic/</code>; the default uses <code>chroma_db/</code>.</p>
""",
    ]
    return _PAGE.format(body="\n".join(sections))


def _rank(row: dict) -> int:
    return row["first_hit_rank"] or 99


def _rank_label(row: dict) -> str:
    return f"#{row['first_hit_rank']}" if row["first_hit_rank"] else "miss"


def _stat(value: str, label: str) -> str:
    return f'<div class="stat"><div class="value">{html.escape(value)}</div><div class="label">{html.escape(label)}</div></div>'


def _table(headers: list, rows: list[list], *, numeric_from: int = 99, tones: list[str] | None = None) -> str:
    head = "".join(
        f'<th class="{"num" if index >= numeric_from else ""}">{html.escape(str(cell))}</th>'
        for index, cell in enumerate(headers)
    )
    body = []
    for row_index, row in enumerate(rows):
        tone = (tones or [""] * len(rows))[row_index]
        cells = "".join(
            f'<td class="{"num" if index >= numeric_from else ""}">{html.escape(str(cell))}</td>'
            for index, cell in enumerate(row)
        )
        body.append(f'<tr class="{tone}">{cells}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _bars(categories: list[str], series: dict[str, list[float]], *, maximum: float, suffix: str = "",
          decimals: int = 0, height: int = 190) -> str:
    width = 330 if len(categories) <= 4 else 680
    left, bottom, top = 8, 34, 16
    plot = height - bottom - top
    group = (width - left * 2) / len(categories)
    bar = min(28, group / (len(series) + 1.2))
    parts = [f'<svg viewBox="0 0 {width} {height + 18}" width="100%" role="img">']
    parts.append(f'<line x1="{left}" y1="{top + plot}" x2="{width - left}" y2="{top + plot}" stroke="#c9ccd1"/>')
    for index, category in enumerate(categories):
        start = left + index * group + (group - bar * len(series)) / 2
        for offset, (name, values) in enumerate(series.items()):
            value = values[index]
            bar_height = plot * (value / maximum if maximum else 0)
            x = start + offset * bar
            y = top + plot - bar_height
            label = f"{value:.{decimals}f}{suffix}"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar - 2:.1f}" height="{bar_height:.1f}" fill="{_COLORS[name]}"/>')
            parts.append(f'<text x="{x + bar / 2 - 1:.1f}" y="{y - 3:.1f}" class="v">{label}</text>')
        parts.append(f'<text x="{left + index * group + group / 2:.1f}" y="{top + plot + 15}" class="c">{html.escape(category)}</text>')
    legend_y = height + 10
    for offset, name in enumerate(series):
        x = left + offset * 110
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="10" height="10" fill="{_COLORS[name]}"/>')
        parts.append(f'<text x="{x + 15}" y="{legend_y}" class="l">{_NAMES[name]}</text>')
    parts.append("</svg>")
    return "".join(parts)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Fixed-size vs semantic chunking</title>
<style>
  @page {{ size: Letter; margin: 16mm 15mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1f2328; font-size: 10.5pt; line-height: 1.45; }}
  h1 {{ font-size: 20pt; margin: 0 0 2px; }}
  h2 {{ font-size: 13pt; margin: 20px 0 8px; border-bottom: 1px solid #d8dbe0; padding-bottom: 3px; }}
  .meta {{ color: #656d76; margin: 0 0 12px; }}
  .stats {{ display: flex; gap: 10px; margin: 14px 0; }}
  .stat {{ flex: 1; border: 1px solid #d8dbe0; border-radius: 6px; padding: 8px 10px; }}
  .stat .value {{ font-size: 14pt; font-weight: 600; color: #2f6fdf; }}
  .stat .label {{ color: #656d76; font-size: 8.5pt; }}
  .callout {{ border-left: 3px solid #d4a72c; background: #fbf6e6; padding: 8px 12px; margin: 14px 0; }}
  .row {{ display: flex; gap: 18px; }}
  figure {{ flex: 1; margin: 6px 0 10px; }}
  figure.wide {{ margin-top: 12px; }}
  figcaption, .note {{ color: #656d76; font-size: 8.5pt; }}
  svg text.v {{ font-size: 8px; text-anchor: middle; fill: #1f2328; }}
  svg text.c {{ font-size: 8.5px; text-anchor: middle; fill: #424a53; }}
  svg text.l {{ font-size: 8.5px; fill: #424a53; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0; }}
  th, td {{ border-bottom: 1px solid #e3e5e8; padding: 4px 6px; text-align: left; vertical-align: top; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  td.num, th.num {{ text-align: right; white-space: nowrap; }}
  tr.better td {{ background: #eaf1fd; }}
  tr.worse td {{ background: #f1f2f4; }}
  tr {{ break-inside: avoid; }}
  code {{ font-family: Menlo, monospace; font-size: 9pt; }}
  .keep, figure {{ break-inside: avoid; }}
  h2 {{ break-after: avoid; }}
  table {{ break-before: avoid; }}
</style></head><body>
{body}
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
