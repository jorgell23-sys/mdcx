"""Checks the reading of a table that the page does not draw.

Most tables in printed material are found from the rules drawn around them,
which costs nothing. What is left is the borderless kind -- a screenshot of a
spreadsheet, a layout held together by alignment -- and guessing at those from
text alignment does not work, which is the one case that earns a model.

Two things about that model matter enough to be fixed here. It reads the
*shape* and never the words: the text keeps coming from the layer in the PDF,
so a cell cannot end up holding something the page does not say. And the shape
it reads comes from pixels, so its column boundaries land approximately -- a
point or two inside a word as often as not. Placing characters one at a time
would split those words across cells and produce a row that is well formed and
wrong throughout, so words are placed whole.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import tables, tatr  # noqa: E402


class Textpage:
    """Characters at known positions, as PDFium reports them."""

    def __init__(self, text: str, start: float = 0.0, bottom: float = 0.0,
                 width: float = 4.0, height: float = 8.0):
        self._chars = []
        x = start
        for character in text:
            self._chars.append((character, x, bottom, x + width, bottom + height))
            x += width

    def count_chars(self) -> int:
        return len(self._chars)

    def get_charbox(self, index, loose=False):
        _, left, low, right, high = self._chars[index]
        return left, low, right, high

    def get_text_range(self, index, count=1):
        return "".join(c[0] for c in self._chars[index:index + count])


def test_a_boundary_inside_a_word_does_not_split_it():
    """The shape comes from pixels, so a boundary lands where it lands.

    "Functions" cut at x=30 would arrive as "F" and "unctions" if characters
    were placed one by one. A word has one centre, so placing it whole keeps
    it whole and puts it in the column its middle falls in.
    """
    page = Textpage("Advanced Functions", start=0.0)     # 4 points a character
    cells = tables.cells_by_word(page, rows=[10.0, 0.0], columns=[0.0, 30.0, 100.0])

    juntas = " ".join(v for v in cells[0] if v)
    assert "Functions" in juntas, f"la palabra se partio: {cells[0]}"
    assert "Advanced" in juntas
    assert not any("unctions" == v for v in cells[0])


def test_words_land_in_the_column_their_middle_falls_in():
    """Two words, two columns, each where it belongs."""
    page = Textpage("ab cd", start=0.0)                  # "ab" 0-8, "cd" 12-20
    cells = tables.cells_by_word(page, rows=[10.0, 0.0], columns=[0.0, 10.0, 30.0])
    assert cells == [["ab", "cd"]]


def test_a_word_outside_the_grid_stays_outside():
    """Placing whole words must not sweep in the margin of the page."""
    page = Textpage("xy", start=-60.0)
    cells = tables.cells_by_word(page, rows=[10.0, 0.0], columns=[0.0, 20.0])
    assert cells == [[""]]


def test_the_shape_is_read_in_the_coordinates_of_the_page():
    """An image counts down from the top; a PDF counts up from the bottom.

    Getting this backwards puts every row of the table in the wrong place, and
    the result still looks like a table.
    """
    escala = tatr.DPI / 72.0
    assert tatr._to_pdf(0.0, escala, 800.0) == pytest.approx(800.0)   # top
    assert tatr._to_pdf(800.0 * escala, escala, 800.0) == pytest.approx(0.0)
    assert tatr._to_pdf(150.0, escala) == pytest.approx(72.0)         # horizontal


def test_the_model_is_asked_for_the_shape_and_not_for_the_words():
    """The words come from the PDF, so nothing can be invented into a cell."""
    fuente = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "convert" / "tatr.py").read_text(encoding="utf-8")
    assert "cells_by_word" in fuente, "las celdas deben salir del texto del PDF"
    for transcribir in ("generate(", "decode(", "ocr"):
        assert transcribir not in fuente, f"no debe transcribir: {transcribir}"


def test_it_reports_whether_it_can_run():
    """A missing extra is a configuration, not a failure."""
    assert isinstance(tatr.available(), bool)


@pytest.mark.skipif(not tatr.available(),
                    reason="needs the tables extra: pip install 'mdcx[tables]'")
def test_the_two_models_answer_different_questions():
    """One is trained on pages and one on tables; swapping them reads prose as columns."""
    assert tatr.DETECTOR != tatr.MODEL
    assert "detection" in tatr.DETECTOR
    assert "structure" in tatr.MODEL
