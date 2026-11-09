"""Command-line interface.

  python -m docuchat ingest sample_docs/brightwave_handbook.pdf
  python -m docuchat ask "How many vacation days do I get?"
  python -m docuchat ask "How many vacation days do I get?" --search-only
"""

from __future__ import annotations

import argparse
import sys

from .answer import LLMNotConfigured, answer_question
from .api import INDEX_PATH
from .ingest import chunk_file
from .search import BM25Index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docuchat")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="add files to the index")
    p_ingest.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=5, help="number of passages to retrieve")
    p_ask.add_argument("--search-only", action="store_true", help="skip the LLM, show passages")

    args = parser.parse_args(argv)
    index = BM25Index.load(INDEX_PATH)

    if args.command == "ingest":
        chunks = list(index.chunks)
        for f in args.files:
            new = chunk_file(f)
            names = {c.source for c in new}
            chunks = [c for c in chunks if c.source not in names] + new
            print(f"{f}: {len(new)} chunks")
        BM25Index(chunks).save(INDEX_PATH)
        print(f"Index now has {len(chunks)} chunks -> {INDEX_PATH}")
        return 0

    hits = index.search(args.question, k=args.k)
    if not hits:
        print("No matching passages. Did you run `ingest` first?")
        return 1
    if not args.search_only:
        try:
            result = answer_question(args.question, hits)
        except LLMNotConfigured as exc:
            print(f"{exc} Showing the best matching passages instead:\n")
        else:
            print(result.text)
            if result.citations:
                print("\nSources:")
                for c in result.citations:
                    print(f'  - {c.source}, page {c.page}: "{c.quote}"')
            return 0
    for h in hits:
        print(f"[{h.score:5.2f}] {h.chunk.source} p.{h.chunk.page}: {h.chunk.text[:160]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
