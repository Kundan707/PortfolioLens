# Portfolio Lens

Ask questions about the earnings of the stocks in your portfolio and the companies they contracted with. The app stores two years of SEC filings in a local Chroma database. Earnings questions use the latest statement, or every statement from the two years when you ask about trends or earlier periods. Contract questions search the full two years.

The symbols come from your brokerage download at `data/portfolio/PortfolioDownload.xls` (set `PORTFOLIO_FILE` to use another path). The file can be CSV or an Apple Numbers document; the reader takes the `Symbol` column and skips `CASH` and `TOTAL`. A held symbol that is not stored yet is pulled from SEC EDGAR on the first question and kept in Chroma. For a symbol that is not in the file, the app says it is not in your portfolio and asks whether to pull it anyway; if you agree, its filings are stored for future questions, and later answers still note that it is not held. From the command line, pass `--pull TICKER` to agree in advance.

## What it does

1. Downloads two years of earnings figures, plus 10-K, 10-Q, and material-agreement text.
2. Marks the latest quarter and the latest fiscal year.
3. Embeds the passages locally and stores them in ChromaDB, tagged by ticker and question type.
4. An earnings question searches only the latest statement, including the revenue-stream breakdown from the filing. A question about trends, comparisons, or an earlier quarter or year gets every statement from the two years, in date order. A contract question searches the two-year set, then summarizes public news from the past year for each named counterparty.
5. Claude writes the answer from those passages and the app shows the sources.

Embeddings stay on your machine. Figures come from filed 10-K and 10-Q data, so they appear after the company files with the SEC.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key from the [Anthropic console](https://console.anthropic.com/settings/keys) in `.env`:

```bash
ANTHROPIC_API_KEY=your-key
```

The first index downloads the local embedding model `BAAI/bge-small-en-v1.5`.

## Run the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Click **Load portfolio earnings** (or just ask about a symbol you hold), choose a symbol or name it in the question, then ask. For example: "What was NVDA diluted EPS in the latest quarter?"

You can also load and ask from the command line:

```bash
python -m rag_agent.earnings
python -m rag_agent.ask "What was AAPL revenue in the latest fiscal year?"
```

## Agent

```text
question
  -> Claude decides to search or answer directly
  -> ChromaDB search of the latest statement, or of two years of contract passages
  -> Claude grades the passages
       relevant -> Claude writes the answer
       off topic -> rewrite the query and search once more
```

`ANTHROPIC_MODEL` defaults to `claude-opus-5-5`.

## Chunking

Contract passages are split into 1,000-character chunks by default. Set `CHUNKING=semantic` to break them where the meaning shifts between sentences instead (slower to index).

Each strategy has its own Chroma instance, so the two sets of chunks never mix: fixed-size uses `chroma_db/` and semantic uses `chroma_db_semantic/`. `CHUNKING` picks which one the app and agent read and write, including tickers pulled on demand. `CHROMA_DIR` overrides both. To compare the two on your portfolio, run:

```bash
python -m rag_agent.chunk_eval          # retrieval only
python -m rag_agent.chunk_eval --llm    # also has Claude answer each question
```

It builds one index per strategy under `data/eval/` and leaves the main library alone.
