from types import SimpleNamespace

from docuchat.answer import build_messages, parse_response
from docuchat.ingest import Chunk, chunk_page
from docuchat.search import BM25Index, Hit, tokenize

# ---------- chunking ----------

def test_chunk_page_respects_max_chars_and_keeps_whole_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_page(text, max_chars=120, overlap_sentences=0)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_chunk_page_overlaps_one_sentence_between_neighbours():
    text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five."
    chunks = chunk_page(text, max_chars=25, overlap_sentences=1)
    for left, right in zip(chunks, chunks[1:], strict=False):
        last_sentence = left.split(". ")[-1]
        assert right.startswith(last_sentence.rstrip("."))


def test_chunk_page_handles_empty_text():
    assert chunk_page("   \n ") == []


# ---------- search ----------

def _index(*texts):
    return BM25Index([Chunk(id=str(i), source="doc.txt", page=i + 1, text=t) for i, t in enumerate(texts)])


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("How many vacation days do I get?") == ["many", "vacation", "day", "get"]


def test_stemming_matches_different_forms_of_a_word():
    assert tokenize("reimbursed expenses") == tokenize("reimburse expense")
    assert tokenize("policies") == ["policy"]


def test_search_ranks_the_relevant_chunk_first():
    index = _index(
        "Employees receive 30 vacation days per year.",
        "Forklifts may only be driven with a licence.",
        "Hotel costs are reimbursed up to 140 euros.",
    )
    hits = index.search("forklift licence")
    assert hits[0].chunk.page == 2


def test_rare_words_outweigh_common_words():
    index = _index(
        "employee employee employee handbook",
        "employee forklift",
        "employee handbook",
    )
    assert index.search("employee forklift")[0].chunk.page == 2


def test_search_returns_nothing_for_unknown_words_and_empty_index():
    assert _index("some text here").search("zebra") == []
    assert BM25Index([]).search("anything") == []


def test_index_save_and_load_roundtrip(tmp_path):
    index = _index("Core hours are 10 to 15.")
    index.save(tmp_path / "index.json")
    loaded = BM25Index.load(tmp_path / "index.json")
    assert loaded.chunks == index.chunks
    assert loaded.search("core hours")[0].chunk.page == 1


# ---------- answer (no network: we test the request we build and the parsing) ----------

HITS = [
    Hit(Chunk("a", "handbook.pdf", 3, "Full-time employees receive 30 vacation days per year."), 5.0),
    Hit(Chunk("b", "handbook.pdf", 5, "Alcohol is never reimbursed."), 2.0),
]


def test_build_messages_sends_one_citable_document_per_chunk():
    content = build_messages("How many vacation days?", HITS)[0]["content"]
    docs = [b for b in content if b["type"] == "document"]
    assert len(docs) == 2
    assert docs[0]["title"] == "handbook.pdf, page 3"
    assert all(d["citations"] == {"enabled": True} for d in docs)
    assert content[-1] == {"type": "text", "text": "How many vacation days?"}


def test_parse_response_maps_citations_back_to_pages():
    citation = SimpleNamespace(document_index=0, cited_text="Full-time employees receive 30 vacation days per year.")
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="You get ", citations=None),
            SimpleNamespace(type="text", text="30 vacation days per year.", citations=[citation, citation]),
        ],
    )
    answer = parse_response(response, HITS)
    assert answer.text == "You get 30 vacation days per year."
    assert len(answer.citations) == 1                      # duplicates removed
    assert (answer.citations[0].source, answer.citations[0].page) == ("handbook.pdf", 3)


def test_parse_response_handles_refusal():
    response = SimpleNamespace(stop_reason="refusal", content=[])
    assert parse_response(response, HITS).refused
