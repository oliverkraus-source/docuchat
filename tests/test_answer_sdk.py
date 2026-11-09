"""Run answer_question through the real Anthropic SDK with a fake HTTP layer.

This checks that the request we send is well-formed for the SDK and that real
SDK response objects (not hand-made stand-ins) are parsed correctly - without
network access or an API key.
"""

import json

import anthropic
import httpx2

from docuchat.answer import MODEL, answer_question
from docuchat.ingest import Chunk
from docuchat.search import Hit

HITS = [
    Hit(Chunk("a", "handbook.pdf", 3, "Full-time employees receive 30 vacation days per year."), 5.0),
    Hit(Chunk("b", "handbook.pdf", 5, "Alcohol is never reimbursed."), 2.0),
]


def test_request_shape_and_citation_parsing_with_real_sdk():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [{
                "type": "text", "text": "You get 30 vacation days per year.",
                "citations": [{
                    "type": "char_location", "document_index": 0, "document_title": "handbook.pdf, page 3",
                    "cited_text": "Full-time employees receive 30 vacation days per year.",
                    "start_char_index": 0, "end_char_index": 55,
                }],
            }],
        })

    client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    answer = answer_question("How many vacation days?", HITS, client=client)

    body = seen["body"]
    assert body["model"] == MODEL
    assert body["fallbacks"] == "default"
    assert seen["beta"] == "server-side-fallback-2026-07-01"
    assert [b["type"] for b in body["messages"][0]["content"]] == ["document", "document", "text"]

    assert answer.text == "You get 30 vacation days per year."
    assert [(c.source, c.page) for c in answer.citations] == [("handbook.pdf", 3)]
