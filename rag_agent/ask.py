"""Ask a question from the command line."""

from __future__ import annotations

import argparse
import sys

from rag_agent.graph import MissingAPIKeyError, answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Claude a question about the indexed PDFs."
    )
    parser.add_argument("question", nargs="+", help="Question to answer")
    args = parser.parse_args(argv)
    try:
        result = answer(" ".join(args.question))
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"The request failed: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    if result.sources:
        print("\nSources")
        for source in result.sources:
            page = f", page {source.page}" if source.page else ""
            print(f"- {source.source}{page}")
    if result.steps:
        print("\nTrace")
        for step in result.steps:
            print(f"- {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
