"""Chat UI for earnings questions about the stocks in the user's portfolio."""

from __future__ import annotations

import html
import traceback

import streamlit as st

from rag_agent.citations import document_name
from rag_agent.config import api_key, chroma_path, get_settings, portfolio_path
from rag_agent.earnings import load_portfolio, normalize_ticker
from rag_agent.graph import (
    AnswerResult,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    Step,
    iter_answer,
)
from rag_agent.portfolio import PortfolioError, portfolio_tickers
from rag_agent.store import clear_library, library_count, list_documents, remove_document

st.set_page_config(page_title="Portfolio Lens", page_icon=":chart_with_upwards_trend:", layout="wide")

MODEL_CHOICES = ["claude-opus-5-5"]


def _holdings() -> tuple[str, ...]:
    try:
        return portfolio_tickers()
    except PortfolioError:
        return ()

EXAMPLES = [
    "What was NVDA diluted EPS in the latest quarter?",
    "What are AAPL's revenue streams in the latest fiscal year?",
    "How has MSFT quarterly revenue and net income changed over the last two years?",
    "Which companies did MSFT sign contracts with in the last two years?",
]


def _init_state() -> None:
    settings = get_settings()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("api_key", api_key())
    st.session_state.setdefault("model", settings.anthropic_model)
    if st.session_state.model not in MODEL_CHOICES:
        st.session_state.model = (
            settings.anthropic_model
            if settings.anthropic_model in MODEL_CHOICES
            else MODEL_CHOICES[0]
        )
    st.session_state.setdefault("notice", None)
    st.session_state.setdefault("allowed_pulls", [])
    st.session_state.setdefault("pending_pull", None)


def _apply_key() -> None:
    import os

    key = (st.session_state.get("api_key") or "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    elif "ANTHROPIC_API_KEY" in os.environ and not api_key():
        os.environ.pop("ANTHROPIC_API_KEY", None)


def _source_fields(source: object) -> tuple[str, int | None, str, str]:
    if isinstance(source, dict):
        return (
            str(source.get("source") or ""),
            source.get("page"),
            str(source.get("excerpt") or ""),
            str(source.get("url") or ""),
        )
    return (
        str(getattr(source, "source", "")),
        getattr(source, "page", None),
        str(getattr(source, "excerpt", "") or ""),
        str(getattr(source, "url", "") or ""),
    )


def _show_sources(sources: list, *, expanded: bool) -> None:
    documents: dict[str, dict] = {}
    articles = []
    filings = []
    for source in sources:
        name, page, excerpt, url = _source_fields(source)
        if url.startswith(("https://", "http://")) and "sec.gov" in url:
            entry = documents.setdefault(url, {"name": name, "excerpts": []})
            if excerpt and excerpt not in entry["excerpts"]:
                entry["excerpts"].append(excerpt)
        elif url.startswith(("https://", "http://")):
            articles.append((name, excerpt, url))
        else:
            filings.append((name, page, excerpt))
    if documents:
        with st.expander("Source documents", expanded=expanded):
            for url, entry in documents.items():
                _external_link(entry["name"], url)
                for excerpt in entry["excerpts"][:2]:
                    st.write(excerpt)
    if articles:
        with st.expander("News articles", expanded=expanded):
            for name, excerpt, url in articles:
                title = excerpt.splitlines()[0] if excerpt else name
                _external_link(title, url)
                st.caption(name)
    if filings:
        with st.expander("Passages used", expanded=expanded and not documents and not articles):
            for name, page, excerpt in filings:
                document = document_name(excerpt)
                if document:
                    label = document if " filed " in name else f"{document} ({name})"
                else:
                    label = f"{name}, page {page}" if page else name
                st.markdown(f"**{label}**")
                st.write(excerpt)


def _external_link(label: str, url: str) -> None:
    st.markdown(
        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>',
        unsafe_allow_html=True,
    )


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            _show_sources(sources, expanded=False)
        steps = message.get("steps") or []
        if steps:
            with st.expander("How the agent got here"):
                for step in steps:
                    st.write(step)


def _run_question(question: str) -> None:
    prior = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
        if item["role"] in {"user", "assistant"}
    ]
    user_message = {"role": "user", "content": question}
    _render_message(user_message)

    steps: list[str] = []
    result: AnswerResult | None = None
    with st.chat_message("assistant"):
        with st.status("Working through the library...", expanded=True) as status:
            try:
                for event in iter_answer(
                    question,
                    history=prior,
                    model=st.session_state.model,
                    allow_pulls=st.session_state.allowed_pulls,
                ):
                    if isinstance(event, Step):
                        steps.append(event.text)
                        status.write(event.text)
                    else:
                        result = event
                if result is None:
                    result = AnswerResult(
                        "The agent finished without an answer.", [], steps
                    )
                status.update(label="Answer ready", state="complete", expanded=False)
            except (MissingAPIKeyError, InvalidAPIKeyError) as exc:
                result = AnswerResult(str(exc), [], steps)
                status.update(label="Needs an API key", state="error", expanded=False)
            except Exception as exc:
                traceback.print_exc()
                message = str(exc)
                if "invalid x-api-key" in message or "authentication_error" in message:
                    message = str(InvalidAPIKeyError())
                else:
                    message = f"The request failed: {exc}"
                result = AnswerResult(message, [], steps)
                status.update(label="The request failed", state="error", expanded=True)
        st.markdown(result.answer)
        if result.sources:
            _show_sources(result.sources, expanded=True)

    st.session_state.messages.append(user_message)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": [
                {
                    "source": source.source,
                    "page": source.page,
                    "excerpt": source.excerpt,
                    "url": source.url,
                }
                for source in result.sources
            ],
            "steps": result.steps or steps,
        }
    )
    if result.pull_ticker:
        st.session_state.pending_pull = {"ticker": result.pull_ticker, "question": question}
        st.rerun()


def _offer_pull() -> None:
    offer = st.session_state.pending_pull
    if not offer:
        return
    ticker = offer["ticker"]
    yes, no, _ = st.columns([1, 1, 4])
    if yes.button(f"Yes, pull {ticker}", type="primary"):
        st.session_state.allowed_pulls.append(ticker)
        st.session_state.pending_question = offer["question"]
        st.session_state.pending_pull = None
        st.rerun()
    if no.button("No, skip it"):
        st.session_state.messages.append(
            {"role": "assistant", "content": f"OK. I did not pull {ticker}, and nothing was stored."}
        )
        st.session_state.pending_pull = None
        st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.header("Earnings library")
        st.caption(
            "Statements are pulled from SEC EDGAR, embedded locally, and stored in Chroma. "
            "Claude only receives the passages needed for a question."
        )
        strategy = get_settings().chunking
        st.caption(f"Chunking: {'semantic' if strategy == 'semantic' else 'fixed-size'} · Chroma at `{chroma_path().name}/`")
        st.text_input(
            "Anthropic API key",
            type="password",
            key="api_key",
            help="Use a key from console.anthropic.com. It starts with sk-ant-.",
        )
        with st.expander("Model"):
            st.selectbox("Claude model", MODEL_CHOICES, key="model")

        st.subheader("Portfolio")
        try:
            holdings = portfolio_tickers()
        except PortfolioError as exc:
            holdings = ()
            st.error(str(exc))
        if holdings:
            st.caption(f"{', '.join(holdings)}  \nFrom {portfolio_path().name}")
        if holdings and st.button("Load portfolio earnings", width="stretch"):
            with st.spinner("Downloading filings from SEC EDGAR…"):
                results = load_portfolio()
            lines = []
            for result in results:
                if result.error:
                    lines.append(f"{result.ticker}: {result.error}")
                else:
                    lines.append(
                        f"{result.ticker}: {result.periods} earnings periods, "
                        f"{result.contract_passages} contract passages"
                    )
            st.session_state.notice = "\n".join(lines)
            st.rerun()

        documents = list_documents()
        st.subheader(f"{library_count()} passages")
        if not documents:
            st.caption("Nothing is indexed yet.")
        grouped: dict[str, list] = {}
        for document in documents:
            ticker = document.source.split()[0]
            grouped.setdefault(ticker, []).append(document)
        for ticker, items in grouped.items():
            held = "" if ticker in holdings else " (not in portfolio)"
            st.markdown(f"**{ticker}**{held}  \n{len(items)} indexed sections")
            if st.button("Remove", key=f"remove-{ticker}"):
                for item in items:
                    remove_document(item.source)
                st.rerun()

        if documents and st.button("Clear library", width="stretch"):
            clear_library()
            st.rerun()


def main() -> None:
    _init_state()
    _apply_key()
    _sidebar()

    st.title("Portfolio Lens")
    st.caption(
        "Earnings questions use the latest statement. A contract question searches the "
        "last two years of filings, then summarizes public news about each named "
        "counterparty from the past year."
    )

    notice = st.session_state.get("notice")
    if notice:
        st.success(notice)
        st.session_state.notice = None

    documents = list_documents()
    if not documents:
        st.info(
            "Load your portfolio earnings, or just ask about a symbol you hold and it will be "
            "pulled from SEC EDGAR. The first load downloads a small local embedding model."
        )

    for message in st.session_state.messages:
        _render_message(message)
    _offer_pull()

    pending = st.session_state.pop("pending_question", None)
    if documents and not st.session_state.messages and pending is None:
        st.write("Try one of these:")
        columns = st.columns(len(EXAMPLES))
        for column, example in zip(columns, EXAMPLES):
            if column.button(example):
                st.session_state.pending_question = example
                st.rerun()

    selected = st.selectbox("Stock symbol", ["From the question", *_holdings()])
    typed = st.chat_input("Ask about earnings for a stock symbol")
    question = pending or typed
    if typed:
        st.session_state.pending_pull = None
    if question:
        symbol = None if selected == "From the question" else normalize_ticker(selected)
        if symbol and symbol not in question.upper():
            question = f"Ticker: {symbol}. {question}"
        _run_question(question)


main()
