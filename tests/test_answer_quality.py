"""Checks what the server says about an answer it cannot give, and the places
where a number meant something other than it appeared to.

The MCP exists so a source can be cited instead of recalled, which only works
if the person asking can tell an answer from the nearest thing to one. Five
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
- The same mixed score went on being printed by `mdcx search` after the MCP
  reply had stopped carrying it. One surface was corrected and the other was
  not, which is the ordinary way a fix half survives.
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
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert '"rank": position' in source
    assert '"score": item.get("score")' not in source, (
        "publicar el score invita a ordenar y filtrar por el, y no es valido")


def test_the_command_line_prints_a_position_and_not_a_score(tmp_path, capsys):
    """The correction has to hold on every surface, not only on the MCP reply.

    Run rather than read: the number was printed by the command, so the command
    is what has to be checked. A lexical result and a dense one arrive in the
    same list with scores of 3.589 and 0.344, which are not on one scale and are
    not what the list is ordered by.
    """
    from mdcx import archive

    received = tmp_path / "src" / "Received"
    received.mkdir(parents=True)
    (received / "printing.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Printing\n\n"
        "Gutenberg built the printing press with movable type in Mainz.\n",
        encoding="utf-8")
    package = tmp_path / "corpus.mdcx"
    archive.pack(tmp_path / "src", package, "k")

    argv = sys.argv
    sys.argv = ["mdcx", "search", str(package), "printing press", "--key", "k"]
    try:
        assert archive.main() == 0
    finally:
        sys.argv = argv

    output = capsys.readouterr().out
    assert "printing" in output, "the passage should have been found"
    assert "score" not in output.lower(), (
        "a score printed over a merged list invites a comparison it cannot support")
    assert "1. " in output, "the position is what the merge establishes"


def test_both_conditions_are_required_because_each_alone_was_wrong():
    """Two measurements, two failure modes, and each one holds the other back.

    How near the best passage comes depends on what is in the collection: one
    corpus answers from 0.57 up and another from 0.64, so a threshold set on
    the first marks nothing on the second. Shipped at 0.45, it never fired.

    How far the best passage stands clear of the fiftieth depends on how many
    passages there are: with a hundred thousand the fiftieth is nearly the
    best, with forty it is nearly the worst. Shipped at 0.25, it marked
    everything.

    Checked here against the edges each was measured on -- the large corpus
    where the absolute number separates and the small one where it does not.
    """
    def mark(sim, clear):
        return sim < mcp_server.NOTHING_NEAR and clear < mcp_server.STANDS_CLEAR

    # Large corpus: answered questions from 0.6427, unanswered up to 0.6320.
    assert not mark(0.6427, 0.2235), "the worst good answer must not be marked"
    assert mark(0.6320, 0.1347), "the best unanswered one must be marked"
    assert mark(0.5452, 0.1150), "recipe for neapolitan pizza dough"

    # Small corpus: answered questions from 0.566, and they stand well clear.
    assert not mark(0.566, 0.331), "a good answer from a small corpus"
    assert mark(0.327, 0.172), "an unrelated query in a small corpus"


def test_relevance_is_judged_against_the_corpus_and_not_against_a_constant():
    """How near anything comes at all depends on the collection.

    On one corpus the questions it answers start at 0.57 and on another at
    0.64, so a threshold on the absolute number marks nothing on the second or
    disparages good answers on the first. A first attempt set at 0.45 never
    fired once outside the corpus it was measured on.

    What travels is the shape of the ranking: an answer stands clear of the
    tail, and a question the corpus knows nothing about does not, because there
    the ordering is only noise.
    """
    assert not hasattr(mcp_server, "NOTHING_CLOSE"), (
        "el umbral absoluto solo no sobrevive un cambio de corpus")
    assert 0.0 < mcp_server.STANDS_CLEAR < 1.0
    assert 0.0 < mcp_server.NOTHING_NEAR < 1.0
    assert mcp_server.TAIL_AT > 1


def test_a_corpus_that_cannot_answer_says_so_without_hiding_anything():
    """Marking, not withholding.

    The nearest passage is worth seeing even when it is not an answer, and a
    corpus that answers in another language must not be swallowed by this.
    """
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert 'answer["warning"]' in source
    assert "found" in source
    # the passages are built before the warning is considered
    assert source.index('"passages"') < source.index('answer["warning"]')


def test_a_package_without_vectors_offers_no_such_number(monkeypatch):
    """Inventing one from BM25 would be the mistake this replaced."""
    monkeypatch.setattr(mcp_server, "_open_packages", lambda: [{"connection": None}])
    monkeypatch.setattr(mcp_server.archive, "_semantic_ready", lambda _: False)
    assert mcp_server._closest_to("cualquier cosa") is None


def test_the_tail_is_what_the_best_passage_is_compared_against():
    """Both numbers are reported, and the judgement rests on the second."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert 'answer["stands_clear"]' in source
    assert "clearance < STANDS_CLEAR" in source, (
        "the warning must rest on the clearance, not on the raw cosine")


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
    paragraph = Lines([(50.0, y, 500.0, y + 10.0) for y in (700.0, 685.0, 670.0, 655.0)])
    assert not tables._rows_are_single_lines(paragraph, [720.0, 640.0, 600.0])


def test_the_cells_of_one_row_are_not_mistaken_for_several_lines():
    """Each cell is reported as its own box, on bases a fraction apart.

    Counting those as separate lines would call every table prose -- which it
    did, taking the tables found in a textbook from twenty-four to one.
    """
    row = Lines([(50.0, 700.0, 150.0, 710.0), (160.0, 700.4, 260.0, 710.4),
                  (270.0, 699.8, 370.0, 709.8)])
    assert tables._rows_are_single_lines(row, [715.0, 695.0, 680.0])


class Index:
    """A package whose index holds a known set of terms."""

    def __init__(self, terms):
        self._terminos = set(terms)

    def __init__2(self):
        pass

    def execute(self, sql, parameters=()):
        if "meta" in sql:                       # el idioma declarado del paquete
            return type("F", (), {"fetchone": lambda _self: ('"en"',)})()
        present = sum(1 for t in parameters if t in self._terminos)
        return type("F", (), {"fetchone": lambda _self: (present,)})()


def test_a_query_the_index_barely_holds_is_answered_by_meaning_alone():
    """Word matching and meaning are merged by rank, which assumes both inform.

    A query written in another language matches a few words by accident -- a
    surname, an abbreviation, a word that exists in both languages -- and those
    few sit at the top of their own list, where rank fusion treats them as the
    equal of the passage that actually answers. Measured over five pairs of
    equivalent questions, the top five held up either way but the first
    position, which is what gets cited, was a quarter worse.
    """
    package = {"connection": Index({"acid", "base", "titration", "ionic", "bond"})}

    # A question in the language of the corpus: the index holds its terms.
    assert mcp_server._mode_for(package, "acid base titration") == "auto"
    assert mcp_server._mode_for(package, "what is an ionic bond") == "auto"

    # One in another language: a single accidental match out of several terms,
    # and the question reads as Spanish. Both are needed.
    assert mcp_server._mode_for(
        package, "una reaccion de acido base y su valoracion") == "semantic"
    assert mcp_server._mode_for(
        package, "cual es la estructura del atomo") == "semantic"


def test_both_signals_are_required_because_each_alone_is_wrong():
    """Neither the share nor the detected language decides on its own.

    The share depends on how much is in the package: on four short documents an
    ordinary English question has a third of its terms indexed, the same as a
    Spanish one on a real corpus. And detection misreads short questions: "how
    does a catalyst work" comes back Portuguese with the confidence a real
    Spanish question comes back Spanish.
    """
    assert 0.0 < mcp_server.LEXICAL_FOOTHOLD < 1.0
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    assert "LEXICAL_FOOTHOLD" in source
    assert "detect_language" in source, (
        "both signals are needed: the share alone condemns legitimate "
        "queries in a small package, where almost nothing is indexed")


def test_a_query_of_only_stopwords_changes_nothing():
    """There is nothing to look up, so there is nothing to decide."""
    package = {"connection": Index({"acid"})}
    assert mcp_server._mode_for(package, "de la que el") == "auto"
