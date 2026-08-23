"""Checks that a document is extracted once however many engines read it.

The pipeline offers the engines in order and treats each as independent: it
hands over a path and takes back Markdown, which is what makes them easy to
chain and to compare. The hybrid is not independent of the native one, though.
It is the native engine plus a second look at the pages that engine could not
settle, and its first line said so:

    pages, meta = _native_pages(path, headings)

So a document that reached the hybrid was extracted twice, with the same
arguments and no state in between, which by construction produces the same
result the second time. Measured at 3.0 ms a page, on 208 of 292 chapters in
one run: about a tenth of its time spent redoing finished work.

The output never changed, which is why nothing caught it. These tests count the
extractions rather than compare the Markdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import engines  # noqa: E402


@pytest.fixture
def counter(monkeypatch):
    """Counts real extractions, and stands in for one."""
    calls: list[tuple] = []

    def bogus(path, headings=None):
        calls.append((str(path), headings))
        pages = [f"# Page one\n\ntext of {Path(path).name}", "page two"]
        return pages, {"engine": "pypdfium2", "pages": 2, "unresolved": [],
                         "tables": 0, "tables_announced": 0, "headings": 0}

    monkeypatch.setattr(engines, "_extract_pages", bogus)
    engines._LAST_EXTRACTION.clear()
    yield calls
    engines._LAST_EXTRACTION.clear()


def test_the_hybrid_does_not_redo_what_the_native_attempt_just_did(counter, tmp_path):
    """The defect, in the order the pipeline produces it."""
    document = tmp_path / "book.pdf"
    engines.native_pdf(document, None)
    engines.hybrid_pdf(document, None)
    assert len(counter) == 1, (
        f"the document was extracted {len(counter)} times for one conversion")


def test_another_document_is_extracted(counter, tmp_path):
    """One entry, so a second document replaces the first rather than joining it."""
    engines.native_pdf(tmp_path / "one.pdf", None)
    engines.native_pdf(tmp_path / "two.pdf", None)
    assert len(counter) == 2
    engines.native_pdf(tmp_path / "one.pdf", None)
    assert len(counter) == 3, "the entry should hold the last document, not every one"


def test_different_headings_are_a_different_extraction(counter, tmp_path):
    """A chapter is extracted with the titles its book carried, and they decide the result."""
    document = tmp_path / "book.pdf"
    engines.native_pdf(document, {1: "Chapter one"})
    engines.native_pdf(document, {1: "Chapter two"})
    assert len(counter) == 2
    engines.native_pdf(document, {1: "Chapter two"})
    assert len(counter) == 2, "the same titles are the same extraction"


def test_what_a_caller_is_given_is_its_own_to_edit(counter, tmp_path):
    """_read_shapes replaces the pages it recovers a table from, in place.

    Handing out the stored list would give the next engine a document the
    previous one had already rewritten -- the same fault as sharing a cache
    between two packages, and as quiet.
    """
    document = tmp_path / "book.pdf"
    first, meta_first = engines._native_pages(document, None)
    first[0] = "REWRITTEN BY THE FIRST ENGINE"
    meta_first["unresolved"].append(99)

    second, meta_second = engines._native_pages(document, None)
    assert len(counter) == 1, "it should still have been extracted only once"
    assert second[0] != "REWRITTEN BY THE FIRST ENGINE", (
        "the second engine was handed the first engine's edits")
    assert meta_second["unresolved"] == [], (
        "the second engine was handed the first engine's pending pages")
