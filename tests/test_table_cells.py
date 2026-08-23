"""Checks that a character lands in exactly one cell of a drawn table.

Cells cannot be cut out as rectangles. Neighbouring cells share the boundary
between them, and PDFium returns every character whose box *intersects* the
rectangle asked for, so a character straddling a boundary comes back in both.
The row still has the right number of columns with every cell filled, so
nothing downstream can tell: a concentration written 2.00x10-6 becomes one
column holding "2" and another holding "2.00 10-6", and it is indexed,
retrieved and quoted with the figure broken in half.

Shrinking the rectangles does not fix it -- a character box reaches several
points into the next cell, and pulling the edge back far enough to exclude it
starts cutting characters that belong. A character has one centre, and that
centre falls in one column.

The edge of the grid is the case that has to be handled rather than ignored:
the outer rule cuts through the first glyph of a line, so its centre lies
outside the grid while the character itself is inside it. Dropping it costs the
first letter of the row -- "Quantum Numbers" arriving as "um Numbers".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import tables  # noqa: E402


class Textpage:
    """A page of characters at known positions, as PDFium would report them.

    Each character is (text, left, bottom, right, top).
    """

    def __init__(self, chars):
        self._chars = chars

    def count_chars(self) -> int:
        return len(self._chars)

    def get_charbox(self, index, loose=False):
        text, left, bottom, right, top = self._chars[index]
        return left, bottom, right, top

    def get_text_range(self, index, count=1):
        return "".join(c[0] for c in self._chars[index:index + count])


def test_a_character_across_a_boundary_is_counted_once():
    """It intersects both cells; its centre is in one of them.

    "b" straddles the boundary at x=10. Read as rectangles it comes back in
    both cells, which is the defect: the row looks well formed and carries a
    letter that the page does not have twice.
    """
    page = Textpage([
        ("a", 1.0, 0.0, 5.0, 8.0),
        ("b", 8.0, 0.0, 12.0, 8.0),     # centre 10.0, exactly on the boundary
        ("c", 14.0, 0.0, 18.0, 8.0),
    ])
    cells = tables._grid_cells(page, rows=[10.0, 0.0], columns=[0.0, 10.0, 20.0])

    assert len(cells) == 1
    joined = "".join(cells[0])
    assert joined.count("b") == 1, f"la b aparece {joined.count('b')} veces: {cells[0]}"
    assert sorted(joined) == ["a", "b", "c"]


def test_a_glyph_cut_by_the_outer_rule_keeps_its_row():
    """The first letter of a line is not lost to the edge of the grid.

    Its centre falls outside the columns because the rule cuts through it, but
    the character is inside the table and the rest of its word is in the first
    cell.
    """
    page = Textpage([
        ("Q", -2.0, 0.0, 2.0, 8.0),    # centre 0.0 is the left edge itself
        ("u", 3.0, 0.0, 6.0, 8.0),
    ])
    cells = tables._grid_cells(page, rows=[10.0, 0.0], columns=[1.0, 10.0])
    assert "Q" in "".join(cells[0]), f"se perdio la Q: {cells[0]}"
    assert "u" in "".join(cells[0])


def test_what_lies_outside_the_grid_stays_outside():
    """Sticking a straddling glyph to the edge must not sweep the margin in."""
    page = Textpage([
        ("x", -40.0, 0.0, -30.0, 8.0),      # well to the left of the table
        ("y", 2.0, 0.0, 6.0, 8.0),
        ("z", 60.0, 0.0, 70.0, 8.0),        # well to the right
    ])
    cells = tables._grid_cells(page, rows=[10.0, 0.0], columns=[0.0, 10.0])
    joined = "".join(cells[0])
    assert "y" in joined
    assert "x" not in joined and "z" not in joined, f"entro el margen: {cells[0]}"


def test_characters_are_placed_in_the_right_row_and_column():
    """Two rows and two columns, one character each, none of them swapped."""
    page = Textpage([
        ("A", 1.0, 11.0, 4.0, 19.0),    # upper left
        ("B", 11.0, 11.0, 14.0, 19.0),  # upper right
        ("C", 1.0, 1.0, 4.0, 9.0),      # lower left
        ("D", 11.0, 1.0, 14.0, 9.0),    # lower right
    ])
    cells = tables._grid_cells(page, rows=[20.0, 10.0, 0.0],
                               columns=[0.0, 10.0, 20.0])
    assert cells == [["A", "B"], ["C", "D"]]


def test_reading_order_survives_inside_a_cell():
    """Characters are walked by index, so a cell reads as it was written."""
    page = Textpage([(c, 1.0 + i, 0.0, 2.0 + i, 8.0)
                     for i, c in enumerate("mono-")])
    cells = tables._grid_cells(page, rows=[10.0, 0.0], columns=[0.0, 40.0])
    assert cells[0][0] == "mono-"


def test_a_page_that_reports_nothing_is_not_a_crash():
    """An empty page yields empty cells of the right shape."""
    cells = tables._grid_cells(Textpage([]), rows=[10.0, 5.0, 0.0],
                               columns=[0.0, 10.0])
    assert cells == [[""], [""]]
