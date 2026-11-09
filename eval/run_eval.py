"""Measure retrieval quality: does search put the right page near the top?

No API key needed - this evaluates the search step, which is where most RAG
systems fail ("garbage in, garbage out": if the right passage is not
retrieved, the model cannot cite it).

Metrics
  hit@k : share of questions where the correct page is in the top k results
  MRR   : mean reciprocal rank - 1.0 if the right page is always ranked first,
          0.5 if it is usually second, and so on

Run:  python eval/run_eval.py            (prints a markdown table)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docuchat.ingest import chunk_file  # noqa: E402
from docuchat.search import BM25Index  # noqa: E402

PDF = ROOT / "sample_docs" / "brightwave_handbook.pdf"
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text())


def evaluate(index: BM25Index, questions=QUESTIONS, k_max: int = 5) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr_total = 0.0
    misses = []
    for item in questions:
        pages = [h.chunk.page for h in index.search(item["q"], k=k_max)]
        rank = pages.index(item["page"]) + 1 if item["page"] in pages else None
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        rr_total += 1 / rank if rank else 0
        if rank != 1:
            misses.append((item["q"], item["page"], pages[:3]))
    n = len(questions)
    return {"hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n, "mrr": rr_total / n, "misses": misses}


def main() -> None:
    configs = [
        ("chunks 600, stemming (default)", dict(max_chars=600), dict(stem=True)),
        ("chunks 600, no stemming", dict(max_chars=600), dict(stem=False)),
        ("chunks 300, stemming", dict(max_chars=300), dict(stem=True)),
        ("chunks 1200, stemming", dict(max_chars=1200), dict(stem=True)),
    ]
    print(f"{len(QUESTIONS)} questions on {PDF.name}\n")
    print("| Config | hit@1 | hit@3 | hit@5 | MRR |")
    print("|---|---|---|---|---|")
    results = {}
    for name, chunk_kw, index_kw in configs:
        index = BM25Index(chunk_file(PDF, **chunk_kw), **index_kw)
        r = evaluate(index)
        results[name] = r
        print(f"| {name} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | {r['mrr']:.2f} |")

    # Breakdown by question type for the default config
    default = BM25Index(chunk_file(PDF))
    print("\n| Question type (default config) | n | hit@1 | hit@5 |")
    print("|---|---|---|---|")
    for qtype in ("keyword", "paraphrase"):
        subset = [q for q in QUESTIONS if q["type"] == qtype]
        r = evaluate(default, subset)
        print(f"| {qtype} | {len(subset)} | {r['hit@1']:.0%} | {r['hit@5']:.0%} |")

    print("\nNot ranked first (default config):")
    for q, expected, got in results["chunks 600, stemming (default)"]["misses"]:
        print(f"  - {q}  (expected p.{expected}, got {got})")


if __name__ == "__main__":
    main()
