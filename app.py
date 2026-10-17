"""Chat UI for the PDF agent."""

from __future__ import annotations

import traceback

import streamlit as st

from rag_agent.config import DATA_DIR, SAMPLE_PDF, api_key, get_settings
from rag_agent.graph import AnswerResult, MissingAPIKeyError, Step, iter_answer
from rag_agent.ingest import ingest_pdf, save_upload
from rag_agent.store import clear_library, library_count, list_documents, remove_document

st.set_page_config(page_title="PDF Agent", page_icon=":page_facing_up:", layout="wide")

MODEL_CHOICES = ["claude-opus-5-5"]

EXAMPLES = [
    "When is the Kestrel's docking window?",
    "What is the backup berth if Pier 7 is occupied?",
    "What is the invoice code for this call?",
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


def _apply_key() -> None:
    import os

    key = (st.session_state.get("api_key") or "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    elif "ANTHROPIC_API_KEY" in os.environ and not api_key():
        os.environ.pop("ANTHROPIC_API_KEY", None)


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            with st.expander("Passages used"):
                for source in sources:
                    page = source.get("page")
                    label = source["source"]
                    if page:
                        label = f"{label}, page {page}"
                    st.markdown(f"**{label}**")
                    st.write(source.get("excerpt") or "")
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
            except MissingAPIKeyError as exc:
                result = AnswerResult(str(exc), [], steps)
                status.update(label="Needs an API key", state="error", expanded=False)
            except Exception as exc:
                traceback.print_exc()
                result = AnswerResult(f"The request failed: {exc}", [], steps)
                status.update(label="The request failed", state="error", expanded=True)
        st.markdown(result.answer)
        if result.sources:
            with st.expander("Passages used", expanded=True):
                for source in result.sources:
                    label = source.source
                    if source.page:
                        label = f"{label}, page {source.page}"
                    st.markdown(f"**{label}**")
                    st.write(source.excerpt)

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
                }
                for source in result.sources
            ],
            "steps": result.steps or steps,
        }
    )


def _sidebar() -> None:
    with st.sidebar:
        st.header("Library")
        st.caption(
            "PDFs are split, embedded locally, and stored in Chroma. "
            "Claude only receives the passages needed for a question."
        )
        st.text_input(
            "Anthropic API key",
            type="password",
            key="api_key",
            help="Stored in this browser session. You can also put ANTHROPIC_API_KEY in .env.",
        )
        with st.expander("Model"):
            st.selectbox("Claude model", MODEL_CHOICES, key="model")

        uploads = st.file_uploader(
            "Add PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Index uploaded PDFs", disabled=not uploads, width="stretch"):
            messages = []
            for upload in uploads or []:
                try:
                    path = save_upload(upload.name, upload.getvalue())
                except ValueError as exc:
                    messages.append(f"{upload.name}: {exc}")
                    continue
                result = ingest_pdf(path)
                if result.error:
                    messages.append(f"{result.source}: {result.error}")
                else:
                    messages.append(
                        f"{result.source}: {result.chunks} passages from {result.pages} pages"
                    )
            st.session_state.notice = "\n".join(messages)
            st.rerun()

        if SAMPLE_PDF.is_file() and st.button("Index the sample brief", width="stretch"):
            result = ingest_pdf(SAMPLE_PDF)
            if result.error:
                st.session_state.notice = result.error
            else:
                st.session_state.notice = (
                    f"Indexed {result.source}: {result.chunks} passages"
                )
            st.rerun()

        documents = list_documents()
        st.subheader(f"{library_count()} passages")
        if not documents:
            st.caption("Nothing is indexed yet.")
        for document in documents:
            page_count = len(document.pages)
            st.markdown(
                f"**{document.source}**  \n{document.chunks} passages · {page_count} pages"
            )
            if st.button("Remove", key=f"remove-{document.source}"):
                remove_document(document.source)
                uploaded = DATA_DIR / document.source
                if uploaded.is_file():
                    uploaded.unlink()
                st.rerun()

        if documents and st.button("Clear library", width="stretch"):
            clear_library()
            st.rerun()


def main() -> None:
    _init_state()
    _apply_key()
    _sidebar()

    st.title("PDF Agent")
    st.caption(
        "Ask a question. The agent searches your PDFs in Chroma, checks that the "
        "passages fit, and Claude Opus 5.5 writes the answer from that context."
    )

    notice = st.session_state.get("notice")
    if notice:
        st.success(notice)
        st.session_state.notice = None

    documents = list_documents()
    if not documents:
        st.info(
            "Index the sample brief, or upload your own PDFs, then ask a question. "
            "The first index downloads a small local embedding model."
        )

    for message in st.session_state.messages:
        _render_message(message)

    pending = st.session_state.pop("pending_question", None)
    if (
        not st.session_state.messages
        and any(document.source == SAMPLE_PDF.name for document in documents)
        and pending is None
    ):
        st.write("Try one of these on the sample brief:")
        columns = st.columns(len(EXAMPLES))
        for column, example in zip(columns, EXAMPLES):
            if column.button(example):
                st.session_state.pending_question = example
                st.rerun()

    typed = st.chat_input("Ask about the indexed PDFs")
    question = pending or typed
    if question:
        _run_question(question)


main()
