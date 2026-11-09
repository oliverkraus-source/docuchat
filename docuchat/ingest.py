"""Turn files into small, searchable chunks that remember where they came from.

Why chunk at all? A language model answers best when it gets the few passages
that matter, not a whole 300-page PDF. And every chunk keeps its source file
and page number, so the final answer can cite exactly where a fact came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str        # unique, e.g. "handbook.pdf#p3#1"
    source: str    # file name
    page: int      # 1-based page number
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_pages(path: str | Path) -> list[str]:
    """Return the text of each page. Plain-text files count as one page."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    if suffix in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")


def _clean(text: str) -> str:
    # PDFs often break lines mid-sentence; join them and squash whitespace.
    text = re.sub(r"-\n(?=\w)", "", text)          # re-join hyphenated words
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def chunk_page(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Split one page into chunks of whole sentences, about `max_chars` long.

    Neighbouring chunks share `overlap_sentences` sentences, so a fact that sits
    on a chunk boundary is still found in one piece.
    """
    sentences = _sentences(_clean(text))
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_file(path: str | Path, max_chars: int = 600, overlap_sentences: int = 1) -> list[Chunk]:
    path = Path(path)
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(read_pages(path), start=1):
        for i, text in enumerate(chunk_page(page_text, max_chars, overlap_sentences)):
            chunks.append(Chunk(id=f"{path.name}#p{page_no}#{i}", source=path.name, page=page_no, text=text))
    return chunks
