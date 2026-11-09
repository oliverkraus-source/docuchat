"""Ask Claude to answer from the retrieved chunks - and prove where each claim came from.

Instead of asking the model to *write* "[page 3]" (which it could get wrong),
each chunk is sent as a separate document with citations enabled. Claude's
response then contains machine-readable citations pointing at the exact
document and the exact sentence it used. We map those back to file + page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic

from .search import Hit

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = (
    "You answer questions using only the documents provided in the user message. "
    "Keep answers short and concrete. If the documents do not contain the answer, "
    "say that you could not find it in the documents instead of guessing."
)


class LLMNotConfigured(RuntimeError):
    """No API credentials are set, so only search (no generated answer) is available."""


def llm_configured() -> bool:
    return any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


@dataclass
class Citation:
    source: str
    page: int
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """One `document` block per chunk (with citations on), then the question."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
            "title": f"{hit.chunk.source}, page {hit.chunk.page}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]
    return [{"role": "user", "content": [*documents, {"type": "text", "text": question}]}]


def parse_response(response, hits: list[Hit]) -> Answer:
    if response.stop_reason == "refusal":
        return Answer(text="The model declined to answer this question.", refused=True)

    parts: list[str] = []
    citations: list[Citation] = []
    seen: set[tuple[str, int, str]] = set()
    for block in response.content:
        if block.type != "text":
            continue                      # skip thinking / fallback blocks
        parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            chunk = hits[c.document_index].chunk
            key = (chunk.source, chunk.page, c.cited_text)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(chunk.source, chunk.page, c.cited_text.strip()))
    return Answer(text="".join(parts).strip(), citations=citations)


def answer_question(question: str, hits: list[Hit], client: anthropic.Anthropic | None = None) -> Answer:
    if not hits:
        return Answer(text="I could not find anything about that in the documents.")
    if client is None:
        if not llm_configured():
            raise LLMNotConfigured("Set ANTHROPIC_API_KEY to get generated answers.")
        client = anthropic.Anthropic()   # reads the key from the environment
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, hits),
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API retries on a recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return parse_response(response, hits)
