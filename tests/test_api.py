import importlib

import pytest
from fastapi.testclient import TestClient

from docuchat.answer import Answer, Citation


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUCHAT_DATA", str(tmp_path))
    import docuchat.api as api
    api = importlib.reload(api)              # pick up the temp data dir

    def fake_answer(question, hits):  # stands in for the real LLM call
        if not hits:
            return Answer("Not found.")
        return Answer("You get 30 days.", [Citation(hits[0].chunk.source, hits[0].chunk.page, "30 days")])

    monkeypatch.setattr(api, "answer_question", fake_answer)
    return TestClient(api.app)


def upload(client, name, text):
    return client.post("/api/documents", files={"file": (name, text.encode(), "text/plain")})


def test_upload_then_ask_returns_answer_with_citation(client):
    text = "Employees get 30 vacation days per year. Core hours are 10 to 15."
    assert upload(client, "policy.txt", text).status_code == 201
    assert client.get("/api/documents").json() == [{"name": "policy.txt", "chunks": 1}]

    data = client.post("/api/ask", json={"question": "How many vacation days?"}).json()
    assert data["answer"] == "You get 30 days."
    assert data["citations"][0]["source"] == "policy.txt"
    assert data["passages"][0]["page"] == 1


def test_reuploading_a_file_replaces_its_chunks(client):
    upload(client, "policy.txt", "Old text about holidays.")
    upload(client, "policy.txt", "New text about vacation.")
    assert client.get("/api/documents").json() == [{"name": "policy.txt", "chunks": 1}]
    assert client.post("/api/ask", json={"question": "holidays please"}).json()["passages"] == []


def test_rejects_unsupported_file_types(client):
    assert upload(client, "virus.exe", "MZ").status_code == 400


def test_upload_cannot_escape_the_data_folder(client, tmp_path):
    upload(client, "../../evil.txt", "Some text here.")
    assert not (tmp_path.parent.parent / "evil.txt").exists()
    assert (tmp_path / "uploads" / "evil.txt").exists()


def test_validates_question_length(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422


def test_falls_back_to_passages_when_no_api_key(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOCUCHAT_DATA", str(tmp_path))
    import docuchat.api as api
    client = TestClient(importlib.reload(api).app)    # real answer_question, not the fake
    upload(client, "policy.txt", "Employees get 30 vacation days per year.")

    res = client.post("/api/ask", json={"question": "vacation days?"})
    assert res.status_code == 200
    assert res.json()["answer"] is None
    assert "ANTHROPIC_API_KEY" in res.json()["error"]
    assert res.json()["passages"][0]["source"] == "policy.txt"
