# Copyright 2026 Jorge Ellena G.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which page of the book a passage came from, once the book is cut up.

A chapter is converted from a PDF extracted from the document, and an extract
starts at page one. So every chapter numbered its pages from one: measured by a
consumer on a 95-page paper, the first three chapters all began at `page 1`, and
the second begins at page 18 of the paper.

No text is lost. What is lost is the one thing the marker carries -- two
passages of the same book both claiming page 3, and a citation that leads
nowhere. For a corpus that exists to be quoted, that is the product.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import convert as convert_module  # noqa: E402
from mdcx.convert.paths import Job  # noqa: E402


def _chapter_job(first, last):
    return Job(source=Path("book.pdf"), rel_source=Path("book.pdf"),
               rel_target=Path("book/02 - Second.md"), kind="pdf", size=1,
               page_range=(first, last), chapter_title="Second",
               chapter_index=2, parent_target=Path("book.md"))


RECORD = {"verification": {"coverage": 1.0, "status": "ok"},
          "digest": "x", "engine": "nativo", "converted_at": "now", "pages": 2}


# --- The markers carry the document's numbering --------------------------------


def test_a_chapter_states_the_pages_of_the_book(self=None):
    """The chapter that begins at page 18 says 18, not 1."""
    extract = "<!-- page 1 -->\nFirst.\n\n<!-- page 2 -->\nSecond.\n"

    renumbered = convert_module._renumber_pages(extract, 18)

    assert "<!-- page 18 -->" in renumbered
    assert "<!-- page 19 -->" in renumbered
    assert "<!-- page 1 -->" not in renumbered


def test_a_chapter_that_opens_the_book_is_left_alone():
    """Its extract and the document agree already, and rewriting identical
    numbers would be a chance to get them wrong."""
    extract = "<!-- page 1 -->\nFirst.\n"

    assert convert_module._renumber_pages(extract, 1) == extract


def test_the_marker_is_matched_however_it_is_spaced():
    """Engines write it, and not all of them write it the same way."""
    assert "<!-- page 20 -->" in convert_module._renumber_pages(
        "<!--page 3-->", 18)
    assert "<!-- page 20 -->" in convert_module._renumber_pages(
        "<!--   page   3   -->", 18)


def test_nothing_else_in_the_text_is_touched():
    """A number that is not a page marker is not a page number."""
    text = "See page 3 of the appendix.\n\n<!-- page 1 -->\nProse.\n"

    renumbered = convert_module._renumber_pages(text, 10)

    assert "See page 3 of the appendix." in renumbered
    assert "<!-- page 10 -->" in renumbered


def test_the_conversion_renumbers_a_chapter():
    """That the rule is reached, and reached with the chapter's own first page
    -- an offset taken from anywhere else would be a plausible wrong answer."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "convert" / "convert.py").read_text(
                  encoding="utf-8")

    assert "_renumber_pages(best_md, job.page_range[0])" in source


# --- And the front matter says it once ------------------------------------------


def test_a_chapter_declares_where_it_starts_in_the_document():
    """So the provenance can be read without counting markers, and so a
    consumer that splits differently can reconcile the two."""
    front = convert_module._front_matter(_chapter_job(18, 29), RECORD)

    assert "first_page: 18" in front
    assert "last_page: 29" in front


def test_a_whole_document_declares_no_such_thing():
    """It has no first page in anything: it is the thing."""
    job = Job(source=Path("book.pdf"), rel_source=Path("book.pdf"),
              rel_target=Path("book.md"), kind="pdf", size=1)

    front = convert_module._front_matter(job, RECORD)

    assert "first_page:" not in front


# --- The index says it is one ---------------------------------------------------


def test_the_index_document_declares_its_type(tmp_path):
    """Asked for because it was thought missing, and it is not: a consumer
    reading only the parent took its small size for an empty conversion and
    raised 252 works as lost. The type is in the front matter, where a reader
    of the file sees it without the record."""
    from mdcx.convert import index

    info = {"document": "book.pdf", "rel_target": Path("book.md"),
            "rel_source": Path("book.pdf"), "from_pdf_outline": True,
            "pages_total": 95, "chapters": 1}
    chapters = [{"markdown_pseudopath": "@/book/01 - One.md", "ok": True,
                 "chapter": {"outline": 1, "title": "One", "pages": [1, 17]},
                 "verification": {"coverage": 1.0, "ref_tokens": 10,
                                  "missing_tokens": 0}}]

    index.write_document_index(info, chapters, tmp_path)

    assert "type: document_index" in (tmp_path / "book.md").read_text(
        encoding="utf-8")
