"""Checks which engine reads a document, and what stops a wrong table.

Reading a page with a vision model costs about a second; extracting its text
costs three milliseconds. That difference only buys something where the page
holds a table, and in a corpus of books nine pages in ten hold none, so the
order the engines are tried in is most of what converting a library costs.

Ordering them is only half of it. Trying the cheap engine first saves nothing
if the expensive one runs anyway, so something has to decide that the cheap
result is enough -- and the thing the cheap engine can quietly get wrong is a
table. A page whose table was read as running text still scores as covered: the
words are all there, in the wrong shape. So the decision rests on what it left
unsettled: a page that announces a table, or is ruled like one, and yielded
none. Counting captions instead was the earlier mistake -- a book that labels
its tables "Figure" announced nothing, so nothing was missing.

The other half is refusing a table that came out wrong. Where a column boundary
falls inside a word the halves land in adjacent cells -- "subtracting" arrives
as "subtractin" and "g numbers" -- and the row still looks well formed. Nothing
downstream would catch it, so it is caught here: the cells of a row must add up
to the line they were cut from.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import convert as C  # noqa: E402
from mdcx.convert import tables  # noqa: E402


def _job(**changes):
    """A conversion job, with only what choosing an engine looks at.

    A chapter carries the book it was cut from and the pages it covers, because
    the headings have to be fetched from there: cutting pages leaves the
    bookmarks behind. A sample carries them for the same reason, scattered
    rather than in a run.
    """
    fields = dict(kind="pdf", is_chapter=False, page_range=None,
                  sample_pages=None, source=Path("book.pdf"))
    fields.update(changes)
    return SimpleNamespace(**fields)


class Textpage:
    """A page that reports the text it holds, which is all a caption needs."""

    def __init__(self, text: str):
        self._text = text

    def get_text_range(self) -> str:
        return self._text


def test_the_cheap_engine_is_tried_first():
    """A vision model is what the order exists to avoid paying for."""
    names = [n for n, _ in C._candidates(_job(), {}, use_docling=True)]
    assert names, "some engine must remain"
    assert names[0] == "nativo"
    assert names.index("nativo") < names.index("docling")


def test_a_document_that_needs_ocr_goes_to_the_model_first():
    """There is nothing to extract from a page with no text layer."""
    names = [n for n, _ in C._candidates(_job(), {"needs_ocr": True}, use_docling=True)]
    assert names[0] == "docling-ocr"


def test_the_cheap_result_stops_the_expensive_engine_when_nothing_is_pending():
    """A document that left no page unsettled and converted cleanly is finished."""
    v = {"status": "ok"}
    score = (1, 5, 1.0, 1000)
    assert C._good_enough("nativo", {"unresolved": []}, v, score) is True


def test_a_page_left_unsettled_sends_the_document_on():
    """Coverage cannot see a missing table: every word is still on the page."""
    v = {"status": "ok"}
    score = (1, 5, 1.0, 1000)
    assert C._good_enough("nativo", {"unresolved": [7]}, v, score) is False
    assert C._good_enough("nativo", {"unresolved": []}, v, score) is True


def test_a_book_that_labels_its_tables_figure_is_not_given_up_on():
    """A count of labels is not a census of tables.

    A book that calls a screenshot of a spreadsheet "Figure 4.2" announces no
    tables at all. Reading that zero as "there are none" left three books in
    six without the model ever looking at them, so what counts now is the page
    left unsettled -- ruled like a table and yielding none -- whatever the
    author called it.
    """
    v = {"status": "ok"}
    score = (1, 5, 1.0, 1000)
    none_labelled = {"tables": 13, "tables_announced": 0, "unresolved": [3, 9]}
    assert C._good_enough("nativo", none_labelled, v, score) is False


def test_an_engine_that_cannot_answer_defers_to_the_model():
    """Silence about tables is not a claim that there were none."""
    v = {"status": "ok"}
    score = (1, 5, 1.0, 1000)
    assert C._good_enough("nativo", {}, v, score) is False


def test_a_bad_conversion_never_stops_the_search():
    """Whatever the tables say, a result that failed is not enough."""
    score = (1, 5, 1.0, 1000)
    meta = {"tables": 4, "tables_announced": 4}
    assert C._good_enough("nativo", meta, {"status": "fail"}, score) is False
    assert C._good_enough("nativo", meta, {"status": "ok"}, (0, 5, 0.2, 10)) is False


def test_a_caption_is_recognised_and_a_mention_is_not():
    """The number and the punctuation are what make it a caption."""
    assert tables.announces_a_table(Textpage("Table 3.1: Ionisation energies"))
    assert tables.announces_a_table(Textpage("Tabla 2 - Resultados"))
    assert tables.announces_a_table(Textpage("Tabelle 4. Messwerte"))
    assert not tables.announces_a_table(Textpage("the table above shows that"))
    assert not tables.announces_a_table(Textpage("a periodic table is useful"))


def test_rules_drawn_as_several_paths_are_one_rule():
    """A rule split into segments must not be counted as several."""
    assert tables._cluster([100.0, 101.5, 300.0]) == pytest.approx([100.75, 300.0])
    assert tables._cluster([]) == []


def test_a_band_needs_two_rules_and_some_height():
    """One rule bounds nothing, and a sliver holds no rows."""
    assert tables._table_band([100.0], 800.0) is None
    assert tables._table_band([100.0, 101.0], 800.0) is None      # too thin
    assert tables._table_band([100.0, 400.0], 800.0) == (100.0, 400.0)


def test_comparing_cells_to_their_line_ignores_only_spacing():
    """Splitting and rejoining moves spaces about; it must not move letters."""
    assert tables._squeeze("Rule: When adding") == tables._squeeze("Rule:Whenadding")
    assert tables._squeeze("subtractin g") != tables._squeeze("subtracting numbers")


def test_no_engine_is_preferred_to_one_that_read_more():
    """The score ranks structure above coverage, so text can be traded for it.

    A result that recovers a table and drops the prose around it wins on
    points while publishing less of the document than the pipeline already
    had. Measured over 214 chapters it happened in 45, and in one the chosen
    result was 16,624 characters shorter than the one already extracted.
    """
    attempts = [{"engine": "nativo", "coverage": 1.0}]
    assert C._covers_less({"coverage": 0.900}, attempts) is True
    assert C._covers_less({"coverage": 0.762}, attempts) is True
    assert C._covers_less({"coverage": 0.997}, attempts) is False   # dentro de la tolerancia
    assert C._covers_less({"coverage": 1.0}, attempts) is False


def test_layout_analysis_is_held_to_the_same_rule():
    """It returns more characters and covers fewer of the document's words."""
    attempts = [{"engine": "nativo", "coverage": 1.0},
                {"engine": "hibrido", "coverage": 0.95}]
    assert C._covers_less({"coverage": 0.990}, attempts) is True


def test_an_engine_that_reads_more_is_not_held_back():
    """The rule is against losing ground, not against improving on it."""
    attempts = [{"engine": "nativo", "coverage": 0.900}]
    assert C._covers_less({"coverage": 1.0}, attempts) is False


def test_a_document_with_nothing_to_compare_against_is_not_rejected():
    """A first attempt, or one with no reference text, has no ground to lose."""
    assert C._covers_less({"coverage": 0.5}, []) is False
    assert C._covers_less({}, [{"engine": "nativo", "coverage": 1.0}]) is False
