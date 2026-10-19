"""Ask a question from the command line."""

from __future__ import annotations

import argparse
import sys

from rag_agent.graph import MissingAPIKeyError, answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Claude a question about the earnings of stocks in your portfolio."
    )
    parser.add_argument("question", nargs="+", help="Question to answer")
    parser.add_argument(
        "--pull",
        action="append",
        default=[],
        metavar="TICKER",
        help="Pull and store this symbol even though it is not in your portfolio",
    )
    args = parser.parse_args(argv)
    try:
        result = answer(" ".join(args.question), allow_pulls=args.pull)
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"The request failed: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    if result.pull_ticker:
        print(f"\nTo pull it, ask again with --pull {result.pull_ticker}")
    if result.sources:
        print("\nSources")
        for source in result.sources:
            page = f", page {source.page}" if source.page and not source.url else ""
            link = f" {source.url}" if source.url else ""
            print(f"- {source.source}{page}{link}")
    if result.steps:
        print("\nTrace")
        for step in result.steps:
            print(f"- {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
