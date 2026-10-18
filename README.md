# Earnings Agent

Ask questions about Magnificent 7 earnings and the companies they contracted with. The app stores two years of SEC filings in a local Chroma database. Earnings questions use the latest statement. Contract questions search the full two years.

The symbols are AAPL, MSFT, GOOGL, AMZN, NVDA, META, and TSLA.

## What it does

1. Downloads two years of earnings figures, plus 10-K, 10-Q, and material-agreement text.
2. Marks the latest quarter and the latest fiscal year.
3. Embeds the passages locally and stores them in ChromaDB, tagged by ticker and question type.
4. An earnings question searches only the latest statement, including the revenue-stream breakdown from the filing. A contract question searches the two-year set, then summarizes public news from the past year for each named counterparty.
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

Click **Load Magnificent 7 earnings**, choose a symbol or name it in the question, then ask. For example: "What was NVDA diluted EPS in the latest quarter?"

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
