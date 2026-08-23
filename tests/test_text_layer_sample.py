"""Checks where a document is sampled when deciding whether it has text.

The decision is whether every page has to be read by a model, which on a book
is the difference between seconds and most of a day. It was taken from the
first three pages.

A book opens with a cover, a blank verso and a title page: no text, no text,
and one line. That is the least representative place in the whole document, and
it is what a publisher varies most.

Measured on real material: three volumes of one physics textbook were sent to
optical recognition because their title page holds 146 characters, while the
English edition of the same book, whose title page holds 200, was read from its
text layer. Both carry some 2,700 characters a page in the body. Fifty-four
characters on the third page decided between reading 581 pages in seconds and
reading them at about 85 seconds each.

Across a corpus of 22 academic books, eight were being sent to optical
recognition with a full text layer, among them one of 1,745 pages -- some forty
hours of work to recover text that was already there. It is what made two
attempts to measure this package hang, which is how it was found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import paths  # noqa: E402


@pytest.mark.parametrize("total", [20, 50, 225, 581, 959, 1745])
def test_the_front_matter_is_never_what_is_sampled(total):
    """Cover, blank verso and title page describe the publisher, not the book.

    Only where there is front matter to skip. A document of six pages is a
    leaflet, and its second page is as good a witness as any.
    """
    chosen = paths._pages_to_sample(total)
    assert min(chosen) > 2, (
        f"a document of {total} pages is judged by page {min(chosen)}, "
        f"which is front matter")


@pytest.mark.parametrize("total", [1, 2, 3, 4, 5])
def test_a_document_shorter_than_the_sample_is_read_whole(total):
    """There is no front matter to skip when there is barely a document."""
    assert paths._pages_to_sample(total) == list(range(total))


@pytest.mark.parametrize("total", [10, 225, 581, 1745])
def test_the_sample_is_spread_rather_than_clustered(total):
    """One region of a book can be atlas, index or bibliography."""
    chosen = paths._pages_to_sample(total)
    assert len(chosen) >= 2
    assert max(chosen) - min(chosen) > total // 3, (
        "the pages looked at sit too close together to describe the document")


@pytest.mark.parametrize("total", [1, 2, 5, 10, 225, 581, 1745, 20000])
def test_every_page_sampled_exists(total):
    """An index past the end is a crash on a document nobody has yet."""
    chosen = paths._pages_to_sample(total)
    assert chosen, "no page at all would be looked at"
    assert all(0 <= i < total for i in chosen)
    assert len(set(chosen)) == len(chosen), "the same page counted twice"


def test_a_title_page_does_not_decide_for_the_body(tmp_path, monkeypatch):
    """The failure itself: empty front matter, and a body full of text.

    Built rather than measured, so it runs without the corpus: three pages of
    front matter as a publisher leaves them, and a body that plainly has a text
    layer. Judged from the front it reads as a scanned document.
    """
    class Page:
        def __init__(self, text):
            self.text = text

    class Document:
        def __init__(self, pages):
            self.pages = pages

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, i):
            return self.pages[i]

        def close(self):
            pass

    body = "x" * 2700
    pages = [Page(""), Page(""), Page("y" * 146)]
    pages += [Page(body) for _ in range(578)]

    from mdcx.convert import pdf as _pdf

    monkeypatch.setattr(_pdf, "open_document", lambda path: Document(pages))
    monkeypatch.setattr(_pdf, "page_text", lambda page: page.text)
    monkeypatch.setattr(_pdf, "count_images", lambda page: 0)

    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4\n")
    job = paths.Job(source=book, rel_source=Path("book.pdf"),
                    rel_target=Path("book.md"), kind="pdf", size=9)

    assert paths.classify_lane(job) == paths.LANE_CPU, (
        "a book with 2,700 characters a page was sent to optical recognition "
        "because its title page is short")
