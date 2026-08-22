"""Checks what the server says about an answer it cannot give, and three
related places where a number meant something other than it appeared to.

The MCP exists so a source can be cited instead of recalled, which only works
if the person asking can tell an answer from the nearest thing to one. Four
things stood in the way, and they share a shape: a figure that looks like a
measurement and is not.

- Each passage carried a `score` drawn from one of two engines. BM25 has no
  upper bound and depends on the corpus it was measured in; cosine runs from
  zero to one and does not. Both arrived in the same field, in a list ordered
  by neither -- merged by rank, because rank is the only thing they agree on.
  A question the corpus could not answer reached 14.65 and one it answered well
  sat at 0.65, so no threshold on that field could ever work.
- Nothing distinguished "here is the answer" from "here is the nearest passage
  there is". The field `found` said three either way.
- A document was sent for optical recognition when a single page of it held no
  text -- a half-title, a blank verso -- which is every book.
- A page ruled for other reasons was read as a table, and its paragraphs came
  back cut into cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import mcp_server  # noqa: E402
from mdcx.convert import extract, tables  # noqa: E402


class Lines:
    """A page reporting the rectangle of each line of text it holds."""

    def __init__(self, boxes):
        self._boxes = boxes

    def count_rects(self):
        return len(self._boxes)

    def get_rect(self, index):
        return self._boxes[index]


def test_a_passage_carries_its_position_and_not_a_score():
    """Position is what the merge establishes; the scores are not comparable."""
    fuente = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert '"rank": position' in fuente
    assert '"score": item.get("score")' not in fuente, (
        "publicar el score invita a ordenar y filtrar por el, y no es valido")


def test_the_threshold_sits_between_what_was_measured():
    """Set from questions a corpus answers and questions it does not.

    The ones it answered came no closer than 0.57; the ones it did not reached
    0.41. A threshold outside that gap either disparages real answers or never
    fires.
    """
    assert 0.41 < mcp_server.NOTHING_CLOSE < 0.57


def test_a_corpus_that_cannot_answer_says_so_without_hiding_anything():
    """Marking, not withholding.

    The nearest passage is worth seeing even when it is not an answer, and a
    corpus that answers in another language must not be swallowed by this.
    """
    fuente = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert 'respuesta["warning"]' in fuente
    assert "found" in fuente
    # the passages are built before the warning is considered
    assert fuente.index('"passages"') < fuente.index('respuesta["warning"]')


def test_a_package_without_vectors_offers_no_such_number(monkeypatch):
    """Inventing one from BM25 would be the mistake this replaced."""
    monkeypatch.setattr(mcp_server, "_open_packages", lambda: [{"connection": None}])
    monkeypatch.setattr(mcp_server.archive, "_semantic_ready", lambda _: False)
    assert mcp_server._closest_to("cualquier cosa") is None


def test_one_blank_page_does_not_send_a_book_to_optical_recognition():
    """Every book has a half-title. Optical recognition is for a scan."""
    assert extract.OCR_SHARE > 0.0
    # one page in sixty is what a half-title looks like
    assert (1 / 60) < extract.OCR_SHARE
    # a document where nothing carries text is what a scan looks like
    assert 1.0 >= extract.OCR_SHARE


def test_rules_around_prose_are_not_read_as_a_table():
    """A row of a table is a line; a block of prose is several.

    Four lines between one pair of rules is a paragraph, and cutting it into
    cells returns shredded sentences that look like a well-formed table.
    """
    parrafo = Lines([(50.0, y, 500.0, y + 10.0) for y in (700.0, 685.0, 670.0, 655.0)])
    assert not tables._rows_are_single_lines(parrafo, [720.0, 640.0, 600.0])


def test_the_cells_of_one_row_are_not_mistaken_for_several_lines():
    """Each cell is reported as its own box, on bases a fraction apart.

    Counting those as separate lines would call every table prose -- which it
    did, taking the tables found in a textbook from twenty-four to one.
    """
    fila = Lines([(50.0, 700.0, 150.0, 710.0), (160.0, 700.4, 260.0, 710.4),
                  (270.0, 699.8, 370.0, 709.8)])
    assert tables._rows_are_single_lines(fila, [715.0, 695.0, 680.0])
