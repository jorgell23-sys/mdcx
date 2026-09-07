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

"""Reaching a word optical recognition misread.

A word transcribed with one letter wrong is a word the index does not contain,
and literal matching can never return it. There is no error to show for it: the
reply is simply empty, and the passage sits in the corpus unreachable.

Optical recognition does not misread letters at random -- it misreads the ones
drawn alike, 0 for O and 1 for l -- so grouping letters by shape puts the misread
word and its original in one bucket. The bucket is far too coarse to answer with,
so it never answers: it proposes, and edit distance decides.

The measurement behind it is not repeated here, having been done on a real
corpus of 188 documents: 225 of 300 misread terms recovered where the search
path recovers 0, at 0.102 ms and 2.96 per cent of package. What is tested here
is that the mechanism does what that measurement assumed, on both of the paths
that can silently do nothing instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import shapekey  # noqa: E402

# Words long enough to be indexed, with the confusions optical recognition
# actually makes: zero for o, one for l, five for s.
PROSE = ("La region metropolitana concentra el conjunto de la poblacion "
         "estudiada durante el periodo.\n\n"
         "El modelo lineal describe el sistema completo de medicion.\n\n"
         "Los resultados del analisis confirman la hipotesis planteada.\n")


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "informe.md").write_text(f"# Informe\n\n{PROSE}", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(folder, target, "k", shapes=True)
    connection, _ = archive.open_package(target, "k")
    return connection


# --- The sieve ----------------------------------------------------------------


def test_letters_drawn_alike_share_a_key():
    """The claim the whole method rests on, over the confusions it is for."""
    for right, wrong in (("region", "regi0n"), ("modelo", "mode1o"),
                         ("sistema", "5istema"), ("lineal", "1ineal")):
        assert shapekey.shape_key(right) == shapekey.shape_key(wrong), \
            f"{right} and {wrong} fall in different buckets"


def test_two_letters_read_as_one_is_outside_this():
    """`rn` for `m` is a real confusion and this does not catch it: the two do
    not share a key, and it is two edits besides.

    Stated because the technique invites the larger claim, and a sieve that is
    believed to catch more than it does is worse than one whose edge is known.
    """
    assert shapekey.shape_key("nurnero") != shapekey.shape_key("numero")


def test_the_shape_alone_is_not_an_answer(corpus):
    """One bucket holds hundreds of words, so it proposes and never decides."""
    same_shape = corpus.execute(
        "SELECT count(*) FROM term_shape WHERE shape = ?",
        (shapekey.shape_key("region"),)).fetchone()[0]

    assert same_shape >= 1
    # And what comes back is only what survives the verification.
    assert shapekey.candidates(corpus, "regi0n") == ["region"]


def test_a_word_too_short_to_mean_anything_is_left_alone(corpus):
    """A three-letter key is shared by hundreds of words: the sieve would
    propose noise and the verification would spend its time refusing it."""
    assert shapekey.MINIMUM_LENGTH >= 4
    assert shapekey.shape_key("de")  # it still has a key
    # but it is neither indexed nor proposed under one
    assert corpus.execute(
        "SELECT count(*) FROM term_shape WHERE length(term) < ?",
        (shapekey.MINIMUM_LENGTH,)).fetchone()[0] == 0
    assert shapekey.candidates(corpus, "de1") == []


def test_one_edit_and_no_more():
    assert shapekey.within_one_edit("region", "regi0n")
    assert shapekey.within_one_edit("region", "regin")
    assert shapekey.within_one_edit("regin", "region")
    assert shapekey.within_one_edit("region", "region")
    assert not shapekey.within_one_edit("region", "reg10n")
    assert not shapekey.within_one_edit("region", "metropolitana")


# --- Only what the corpus lacks -----------------------------------------------


def test_a_term_the_corpus_has_is_not_widened(corpus):
    """Widening it would trade an exact match for candidates nobody asked for."""
    assert shapekey.expand(corpus, ["modelo"]) == {}


def test_the_terms_are_normalised_before_they_are_looked_up(corpus):
    """`df` holds folded keys, so an unfolded term is absent from it for the
    wrong reason -- reported unknown, then given a key over characters the table
    does not hold, which matches nothing. The whole thing would do nothing at
    all, quietly."""
    assert shapekey.expand(corpus, ["MODELO"]) == {}
    assert shapekey.expand(corpus, ["Regi0n"]) == {"regi0n": ["region"]}


def test_a_package_without_the_index_answers_as_it_did(tmp_path):
    """Every package written before this has none, and must keep working."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")
    archive.pack(folder, tmp_path / "plain.mdcx", "k")   # no shapes
    connection, _ = archive.open_package(tmp_path / "plain.mdcx", "k")

    assert not shapekey.has_index(connection)
    assert shapekey.expand(connection, ["regi0n"]) == {}
    assert shapekey.candidates(connection, "regi0n") == []
    assert archive.lexical_query(connection, "region metropolitana", limit=3)


# --- Through the search path --------------------------------------------------


def test_a_misread_word_reaches_its_passage(corpus):
    """What literal matching cannot do, and does not report failing to do."""
    found = archive.lexical_query(corpus, "regi0n", limit=3)

    assert found
    assert "region metropolitana" in found[0]["passage"]


def test_it_runs_only_where_the_answer_would_be_empty(corpus):
    """The boundary, and it is narrower than it looks.

    Terms are matched with OR, so one word of a question that the corpus does
    contain is enough to return passages -- and then the sieve never runs, even
    though another word of that same question was misread. The answer is not
    empty, so there is nothing being lost; what is lost is the chance to say
    that one of the words was read as something else.

    Widening earlier would mean widening queries that already work, trading
    exact matches for candidates nobody asked for.
    """
    notes: dict = {}
    archive.lexical_query(corpus, "regi0n metropolitana", limit=3, notes=notes)
    assert "read_as" not in notes, "it widened a query that already matched"

    notes = {}
    archive.lexical_query(corpus, "p0blacion c0njunto", limit=3, notes=notes)
    assert set(notes["read_as"]) == {"p0blacion", "c0njunto"}


def test_the_scoring_terms_are_widened_too(corpus):
    """The half of the integration that would silently return nothing.

    The passage found this way holds the word that was written, not the word
    that was asked about, so a frequency count over the original terms alone
    comes back empty and every result is dropped one loop later. Widening the
    expression without widening what is counted looks exactly like the sieve
    finding nothing.
    """
    widened = archive.lexical_query(corpus, "mode1o", limit=3)

    assert widened, "the query was widened and the results were then discarded"


def test_a_query_that_matches_is_not_touched(corpus):
    """The sieve runs only where the answer is already empty."""
    notes: dict = {}
    archive.query(corpus, "region metropolitana", limit=3, notes=notes)

    assert "read_as" not in notes


def test_a_query_about_nothing_still_returns_nothing(corpus):
    """Widening must not turn an honest empty answer into noise."""
    assert archive.lexical_query(corpus, "termodinamica cuantica", limit=3) == []


def test_what_was_read_as_what_is_reported(corpus):
    """A passage found under another spelling is not a literal match, and
    presenting it as one spends the thing that makes the package worth having:
    that a citation carries the words of the document."""
    notes: dict = {}
    archive.query(corpus, "regi0n", limit=3, notes=notes)

    assert notes["read_as"] == {"regi0n": ["region"]}


# --- Adding it to a package that was written without it -----------------------


def test_the_index_can_be_added_afterwards(tmp_path):
    """A package whose source material is gone can still gain it: the terms
    come from `df`, which the package already carries."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")
    target = tmp_path / "later.mdcx"
    archive.pack(folder, target, "k")
    before, _ = archive.open_package(target, "k")
    assert not shapekey.has_index(before)

    archive.add_shapes(target, "k")

    after, header = archive.open_package(target, "k")
    assert header["_intact"], "the package no longer verifies against its header"
    assert shapekey.has_index(after)
    assert archive.lexical_query(after, "regi0n", limit=3)


def test_a_signed_package_refuses_rather_than_coming_back_unsigned(tmp_path):
    private, _ = archive.generate_signing_key()
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")
    target = tmp_path / "signed.mdcx"
    archive.pack(folder, target, "k", signing_key=private)

    with pytest.raises(ValueError, match="signed"):
        archive.add_shapes(target, "k")

    archive.add_shapes(target, "k", signing_key=private)
    assert archive.read_header(target)["signature"]


# --- What it costs, and that it is asked for ----------------------------------


def test_the_index_is_not_built_unless_it_is_asked_for(tmp_path):
    """It makes the package bigger and buys nothing on a corpus that never went
    through optical recognition. Charging every corpus for what serves some is
    what this project keeps declining to do."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")

    quiet = archive.pack(folder, tmp_path / "q.mdcx", "k")
    asked = archive.pack(folder, tmp_path / "a.mdcx", "k", shapes=True)

    assert not quiet.get("shape_terms")
    assert asked["shape_terms"] > 0


# --- What the audit found -----------------------------------------------------


def test_the_reply_carries_what_was_read_as_what(tmp_path, monkeypatch):
    """The safeguard was dead: the note reached the engine and stopped there.

    `search_packages` collects a note per package and merges it, and the merge
    carried only the fields it knew about. So the server returned a passage
    found under a *different word* and presented it as a literal match -- the
    exact failure this field exists to prevent, and invisible from outside.
    """
    import asyncio
    import importlib
    import json

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")
    target = tmp_path / "served.mdcx"
    archive.pack(folder, target, "k", shapes=True)
    monkeypatch.setenv("MDCX_FILE", str(target))
    monkeypatch.setenv("MDCX_KEY", "k")

    from mdcx import mcp_server
    importlib.reload(mcp_server)
    output = asyncio.run(mcp_server.create_server().call_tool(
        "search", {"query": "regi0n"}))
    assert not output.is_error, output.content
    answer = json.loads(output.content[0].text)

    assert answer["found"] > 0
    assert answer["read_as"] == {"regi0n": ["region"]}

    # And absent where nothing was substituted, so the field means something.
    plain = asyncio.run(mcp_server.create_server().call_tool(
        "search", {"query": "region"}))
    assert "read_as" not in json.loads(plain.content[0].text)


def test_nothing_is_claimed_when_the_widened_match_found_nothing(corpus):
    """Proposing a candidate is not finding a passage.

    The second match can still come back empty -- `direction` scopes it while
    the vocabulary it drew from does not -- and a note surviving that would tell
    a reader that passages the semantic engine found were read under another
    spelling, which they were not.
    """
    notes: dict = {}
    archive.lexical_query(corpus, "regi0n", limit=3, only="SENT", notes=notes)

    assert "read_as" not in notes


def test_only_the_words_that_were_asked_are_widened(corpus):
    """The glossary appends cross-language equivalents, absent from a
    monolingual corpus for a reason that has nothing to do with transcription.
    Widening those would report, as a word of the question the corpus lacks, a
    word nobody typed."""
    notes: dict = {}
    archive.lexical_query(corpus, "regi0n", limit=3, notes=notes)

    assert set(notes.get("read_as", {})) <= {"regi0n"}


def test_an_index_of_another_version_is_treated_as_none(corpus):
    """Keys built under one table line up on nothing under another, and the
    failure is the quiet kind: an empty answer with no error to show."""
    assert shapekey.has_index(corpus)

    corpus.execute("UPDATE meta SET value = ? WHERE key = 'shape_version'",
                   (str(shapekey.SHAPE_VERSION + 1),))

    assert not shapekey.has_index(corpus)
    assert shapekey.expand(corpus, ["regi0n"]) == {}


def test_the_searchable_copy_is_written_once(tmp_path):
    """It was computed at insert, thrown away, computed again, and the table
    rewritten three times -- for a value nothing reads in between."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(f"# A\n\n{PROSE}", encoding="utf-8")
    archive.pack(folder, tmp_path / "c.mdcx", "k")
    connection, _ = archive.open_package(tmp_path / "c.mdcx", "k")

    kept = connection.execute(
        "SELECT count(*) FROM passage WHERE search_text IS NOT NULL").fetchone()[0]

    assert kept == 0, "the searchable copy is still being stored"
    assert archive.lexical_query(connection, "region metropolitana", limit=3)
