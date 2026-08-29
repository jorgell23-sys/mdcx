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


# --- The second signal has to be inside the range it measures ---------------
#
# Eighteen questions over a corpus of five statistics books, ten the corpus
# answers and eight deliberately unrelated. Measured on 1.8.0, cosine of the
# best passage and how far it rises out of the tail.

STATISTICS_BANK = [
    # (closeness, clearance, the corpus answers it)
    (0.6511, 0.0924, True),
    (0.7093, 0.1108, True),
    (0.6337, 0.1501, True),   # marked in 1.8.0, by 13 ten-thousandths
    (0.6757, 0.1515, True),
    (0.6586, 0.1087, True),
    (0.7176, 0.1225, True),
    (0.6924, 0.0856, True),
    (0.6080, 0.1046, True),   # marked in 1.8.0
    (0.6495, 0.1187, True),
    (0.6516, 0.1007, True),
    (0.4658, 0.0775, False),
    (0.4978, 0.1024, False),  # the unrelated question that rises furthest
    (0.4464, 0.0793, False),
    (0.4824, 0.0605, False),
    (0.4566, 0.0753, False),
    (0.4535, 0.0395, False),
    (0.3291, 0.0000, False),
    (0.4467, 0.0656, False),
]


def _warns(closeness, clearance):
    return (closeness < mcp_server.NOTHING_NEAR
            and clearance < mcp_server.STANDS_CLEAR)


def test_no_unrelated_question_escapes_the_warning():
    """What the warning is for, and the constraint any change has to keep."""
    escaped = [(c, cl) for c, cl, answered in STATISTICS_BANK
               if not answered and not _warns(c, cl)]

    assert not escaped, f"unrelated questions came back unmarked: {escaped}"


def test_the_two_banks_overlap_so_no_clearance_threshold_separates_them():
    """Why STANDS_CLEAR was not simply lowered, recorded so it is not retried.

    On the statistics corpus every clearance falls below 0.1515, so 0.25 never
    fails there and the second signal looks dead. Lowering it into that range
    is the obvious repair and it does not hold: on the corpus this constant was
    measured on, an unrelated question clears 0.172 and an answered one clears
    0.331. A threshold under 0.172 lets that unrelated question through.
    """
    unrelated_elsewhere = 0.172        # measured, and unrelated
    answered_elsewhere = 0.331         # measured, and answered

    highest_here = max(cl for _, cl, _ in STATISTICS_BANK)
    would_rescue = [cl for _, cl, answered in STATISTICS_BANK
                    if answered and cl > unrelated_elsewhere]

    assert highest_here < unrelated_elsewhere, (
        "the statistics bank now reaches the other corpus's unrelated range, "
        "so this argument needs re-measuring")
    assert not would_rescue, (
        "a threshold that keeps the unrelated question out rescues nothing "
        "here, which is the finding: the ranges overlap between corpora")
    assert answered_elsewhere > unrelated_elsewhere, (
        "on that corpus the signal did separate, which is why it exists")


def test_the_question_missed_by_a_hair_is_still_marked_and_that_is_known():
    """0.6337 against 0.635: thirteen ten-thousandths, and still wrong.

    Recorded rather than fixed. The corpus answers it out of the chapter where
    the subject is explained, five passages of five, and the warning still
    fires. Nothing here can fix it without breaking the other bank; what it
    needs is a threshold measured from the corpus rather than carried into it.
    """
    assert _warns(0.6337, 0.1501), (
        "this now passes -- if the thresholds were changed, the report that "
        "measured this case should be re-run rather than this test relaxed")


# --- A threshold the corpus measures about itself ---------------------------
#
# Both constants failed the same way: they describe a collection they cannot
# see. Measured on packages built for this, the same questions reach 0.51 on one
# corpus and 0.55 on another, so a fixed cut lands inside the answered range of
# one and below the other's.
#
# What packing can measure is how near the corpus comes to a question it does
# answer: a passage used as a query, with the query prefix, standing in for one.


def test_a_package_without_the_measurement_answers_as_it_always_did():
    """No regression for a package built before this existed.

    An absent measurement is not a low one, so the two constants decide exactly
    as they did rather than a missing number being read as zero.
    """
    for closeness, clearance, _ in STATISTICS_BANK:
        old = closeness < mcp_server.NOTHING_NEAR and clearance < mcp_server.STANDS_CLEAR
        assert mcp_server._nothing_near(closeness, clearance, None) is old


def test_the_measured_reach_rescues_the_questions_the_constant_condemned():
    """The two the report measured, on a corpus whose reach is known.

    Built and packed for this: twenty-four passages of statistics, reach 0.8185.
    The two questions are the same ones -- mean, median and mode, in both
    languages -- and the absolute threshold marked both.
    """
    reach = 0.8185
    for closeness in (0.5534, 0.5934):
        assert closeness < mcp_server.NOTHING_NEAR, "premise: the constant marks it"
        assert not mcp_server._nothing_near(closeness, 0.10, reach), (
            f"a question the corpus answers at {closeness} is still marked")


def test_no_unrelated_question_escapes_the_measured_reach():
    """Measured against the same package: unrelated questions reach 0.30-0.33."""
    reach = 0.8185
    for closeness in (0.3136, 0.3290, 0.3056):
        assert mcp_server._nothing_near(closeness, 0.05, reach), (
            f"an unrelated question at {closeness} came back unmarked")


def test_the_same_question_is_judged_against_the_corpus_it_is_asked_of():
    """What no constant could do, and the whole point of measuring the corpus.

    The statistics questions against a corpus of botany reach 0.31-0.39, and
    that corpus reaches 0.8327 of its own. The question is unanswerable there
    and answerable in the other, and the judgement follows the corpus.
    """
    assert mcp_server._nothing_near(0.3316, 0.05, 0.8327), (
        "a question this corpus cannot answer was let through")
    assert not mcp_server._nothing_near(0.5534, 0.10, 0.8185), (
        "the same question, on the corpus that answers it, is still marked")


def test_the_share_sits_between_what_was_measured():
    """The cut is between the worst answered and the best unrelated, not on one."""
    worst_answered = 0.5534 / 0.8185
    best_unrelated = 0.3883 / 0.8327

    assert best_unrelated < mcp_server.ANSWERS_AT_SHARE < worst_answered, (
        f"the share {mcp_server.ANSWERS_AT_SHARE} is outside the measured gap "
        f"{best_unrelated:.4f} to {worst_answered:.4f}")


# --- A package can be told the questions it exists to answer -----------------
#
# Calibrating against passages used as probes is only as good as passages
# resembling questions, and on some corpora they do not. A catalogue of
# 80,844 records -- all back-cover blurbs, all sharing a rhetorical shape --
# calibrated at 0.7588 where a corpus of books calibrates at 0.580. At 0.7588
# the warning fires on questions the catalogue answers well: the same false
# positive three reports have measured, returning through the other side and
# for the same underlying reason, which is calibrating without questions.


def test_the_two_calibrations_mean_different_things():
    """One is an estimate of the threshold; the other is the threshold."""
    assert mcp_server.ANSWERS_AT_SHARE < mcp_server.ASKED_MARGIN <= 1.0, (
        "a reach taken from the questions themselves needs no discount for "
        "being an estimate, and scaling it like one puts the cut far below "
        "anything that was measured")


def test_the_margin_only_absorbs_the_rounding():
    """Not 1.0, which is where the arithmetic points and where it fails.

    The reach is stored rounded to four places and the cosine computed at query
    time falls either side of it, so a cut set exactly at the worst declared
    question marks that very question.
    """
    worst_declared = 0.6281
    assert worst_declared * mcp_server.ASKED_MARGIN < worst_declared, (
        "the cut sits at the declared question and will mark it")
    assert worst_declared * mcp_server.ASKED_MARGIN > worst_declared * 0.9, (
        "the cut drifted well below what was declared, which is what "
        "calibrating from questions was meant to avoid")


def test_a_question_the_package_was_given_is_not_marked():
    """Measured on a package of catalogue records built for this.

    Probes drawn from those records reach 0.695 between themselves while the
    question they exist to answer reaches 0.628 -- the shape of the defect this
    corrects, in miniature.
    """
    reach_from_questions = 0.6281
    assert not mcp_server._nothing_near(0.6281, 0.10, reach_from_questions,
                                    from_questions=True), (
        "the question the package was told about is marked as one nothing in "
        "it is about")
    assert not mcp_server._nothing_near(0.6786, 0.10, reach_from_questions,
                                    from_questions=True)


def test_unrelated_questions_are_still_marked_when_calibrated_from_questions():
    """The cut moved up, so this is the side that had to be checked."""
    reach_from_questions = 0.6281
    for closeness in (0.4032, 0.2380):
        assert mcp_server._nothing_near(closeness, 0.05, reach_from_questions,
                                from_questions=True), (
            f"an unrelated question at {closeness} came back unmarked")
