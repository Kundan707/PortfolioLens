# PDF Agent

Ask questions about your PDFs. The app indexes them in a local Chroma database, and a LangGraph agent uses Claude Opus 5.5 to search that library, check the passages, and write an answer.

## What it does

1. Reads each PDF and splits it into passages.
2. Embeds those passages locally and stores them in ChromaDB.
3. When you ask a question, the model decides whether to search.
4. Chroma returns the closest passages. The model grades them.
5. If they miss the question, the agent rewrites the search and tries once more.
6. Claude writes the answer from the passages and the app shows the sources.

Embeddings stay on your machine. The model receives the question and the retrieved passages, not the whole file.

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

Index the included Northwind Harbor sample, or upload your own text PDFs, then ask a question. Scanned PDFs without a text layer are not supported.

You can also use the command line:

```bash
python -m rag_agent.sample
python -m rag_agent.ingest sample_docs
python -m rag_agent.ask "When is the Kestrel's docking window?"
```

## Agent

```text
question
  -> Claude decides to search or answer directly
  -> ChromaDB similarity search
  -> Claude grades the passages
       relevant -> Claude writes the answer
       off topic -> rewrite the query and search once more
```

`ANTHROPIC_MODEL` defaults to `claude-opus-5-5`.
