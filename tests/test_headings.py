"""Checks that a chapter keeps the section titles its book already carried.

Extracting text gives paragraphs and nothing else: no headings, no lists. That
is not a cosmetic loss. The titles an author writes -- "Osmosis and Osmotic
Pressure of Solutions" -- are among the best answers a search over the corpus
can return, and a chapter without them is a wall of prose with nothing to cite.

It is also what decided which engine converted the document. A result with no
structural marks was rejected as incomplete, so a chapter of running prose --
correctly extracted, covering every word -- could never be accepted, and every
document without tables went to layout analysis however well the cheap path had
done. Measured on a real run: 81 per cent of chapters, at a fifth of a page per
second against fifty.

The titles do not have to be inferred. A PDF carries them as bookmarks, with
the hierarchy included. The catch is that a chapter is converted from pages cut
out of the book, and cutting pages leaves the bookmarks behind -- so they have
to be fetched from the book and translated to where the pages ended up.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import pdf  # noqa: E402

HEADING = re.compile(r"(?m)^#{1,6} ")


def test_bookmarks_are_translated_to_where_the_pages_ended_up():
    """A bookmark on page 42 of the book is on page 3 of a chapter starting at 40."""
    marcas = [(1, "Chapter 3", 40), (2, "Solutions", 42), (2, "Osmosis", 45),
              (1, "Chapter 4", 100)]
    original = pdf.outline
    try:
        pdf.outline = lambda _: marcas
        found = pdf.outline_for_range(Path("libro.pdf"), 40, 60)
    finally:
        pdf.outline = original

    assert found == {1: [(1, "Chapter 3")], 3: [(2, "Solutions")], 6: [(2, "Osmosis")]}
    assert 100 not in found, "lo de otro capitulo no debe entrar"


def test_a_range_with_no_bookmarks_is_empty_rather_than_wrong():
    """A book without a table of contents is a supported case, not a failure."""
    original = pdf.outline
    try:
        pdf.outline = lambda _: []
        assert pdf.outline_for_range(Path("x.pdf"), 1, 10) == {}
        pdf.outline = lambda _: (_ for _ in ()).throw(RuntimeError("roto"))
        try:
            pdf.outline_for_range(Path("x.pdf"), 1, 10)
        except RuntimeError:
            pass        # el llamador decide; lo que no debe es inventar titulos
    finally:
        pdf.outline = original


def test_the_headings_reach_the_markdown():
    """What is fetched has to come out as headings, at the right depth."""
    from mdcx.convert import engines

    class Pagina:
        def get_textpage(self):
            return self
        def get_text_range(self, *a):
            return "cuerpo del capitulo"
        def count_chars(self):
            return 0
        def count_rects(self):
            return 0
        def get_width(self):
            return 600.0
        def get_height(self):
            return 800.0
        def get_objects(self, max_depth=1):
            return []

    class Documento(list):
        def close(self):
            pass

    original_open = engines_open = None
    from mdcx.convert import pdf as _pdf
    original_open, original_par = _pdf.open_document, _pdf.page_paragraphs_fast
    try:
        _pdf.open_document = lambda _: Documento([Pagina(), Pagina()])
        _pdf.page_paragraphs_fast = lambda _: ["cuerpo del capitulo"]
        paginas, meta = engines._native_pages(
            Path("cap.pdf"), {1: [(1, "Chapter 3"), (2, "Solutions")], 2: [(3, "Osmosis")]})
    finally:
        _pdf.open_document, _pdf.page_paragraphs_fast = original_open, original_par

    entero = "\n".join(paginas)
    assert meta["headings"] == 3
    assert len(HEADING.findall(entero)) == 3
    assert "# Chapter 3" in entero
    assert "## Solutions" in entero
    assert "### Osmosis" in entero


def test_a_chapter_of_prose_is_no_longer_rejected_as_structureless():
    """The score that decided the engine counted marks, and prose has none.

    With the headings restored the same chapter carries structure, which is
    what lets a correct cheap conversion be accepted instead of every document
    without tables going to layout analysis.
    """
    from mdcx.convert import convert as C

    sin_titulos = "parrafo uno\n\nparrafo dos\n"
    con_titulos = "# Chapter 3\n\nparrafo uno\n\n## Solutions\n\nparrafo dos\n"
    v = {"status": "ok", "coverage": 1.0}

    assert C._score(sin_titulos, v)[1] == 0
    assert C._score(con_titulos, v)[1] > 0

    # A result with no structural marks is not accepted however much of the
    # document it covered: a chapter returned as a wall of prose is exactly
    # what the expensive engine is for.
    assert C._good_enough("hibrido", {"unresolved": []}, v,
                          C._score(sin_titulos, v), []) is False
    assert C._good_enough("hibrido", {"unresolved": []}, v,
                          C._score(con_titulos, v), []) is True
