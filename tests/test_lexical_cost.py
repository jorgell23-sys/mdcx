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

"""What a lexical query spent redoing what SQLite had already done.

Profiled by a consumer on one package of 405,292 passages: a query took 2.50 s,
of which the FTS5 match -- the actual retrieval -- was 0.34 s. The rest was
Python: reading `df` whole to use five of its rows, tokenising every candidate
passage twice for counts the index already held, and scanning
`document.normalized_text` -- the corpus again -- to reorder documents that were
already in hand.

None of what follows changes a ranking. The counts are the same counts, taken
from where they were recorded rather than recomputed, which is why the tests
that matter here compare results rather than times.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402


def _corpus(tmp_path, documents=40, vocabulary=3000):
    random.seed(41)
    words = [f"term{i:04}" for i in range(vocabulary)]
    common = "teorema demostracion lema espacio funcion continua conjunto".split()
    folder = tmp_path / "md"
    folder.mkdir()
    for i in range(documents):
        blocks = []
        for _ in range(6):
            block = ([random.choice(words) for _ in range(40)]
                     + [random.choice(common) for _ in range(8)])
            random.shuffle(block)
            blocks.append(" ".join(block))
        (folder / f"d{i:03}.md").write_text(
            f"# Doc {i}\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(folder, target, "k")
    return archive.open_package(target, "k")[0]


QUERIES = ("teorema demostracion espacio",
           "funcion continua conjunto",
           "lema espacio funcion continua conjunto teorema")


def _answers(connection):
    return [[(x["document"], x["score"], tuple(x["terms"]))
             for x in archive.query(connection, q, limit=6)] for q in QUERIES]


# --- The counts come from the index, and are the same counts -------------------


def test_reading_the_counts_from_the_index_changes_no_ranking(tmp_path,
                                                              monkeypatch):
    """The whole safety of it. FTS5 recorded every occurrence when the package
    was built; asking it is a lookup where re-reading the text is work
    proportional to the text. What it must not do is answer differently."""
    connection = _corpus(tmp_path)

    from_index = _answers(connection)
    monkeypatch.setattr(archive, "_occurrences", lambda *a, **k: None)
    from_text = _answers(connection)

    assert from_index == from_text, (
        "the index and the text disagree about what the passages hold")


def test_a_package_without_the_counts_is_still_answered(tmp_path, monkeypatch):
    """Packages written before the token count existed do not carry it, and
    they are read by tokenising the candidates as before. The column is the
    fast path, not the only one."""
    connection = _corpus(tmp_path)
    monkeypatch.setattr(archive, "_has_token_counts", lambda c: False)
    archive._COLUMN_CACHE.clear()

    assert archive.query(connection, "teorema demostracion", limit=4)


def test_the_stored_length_is_the_one_the_scoring_rule_uses(tmp_path):
    """Counted by the same rule that built `df`, so BM25 normalises by the same
    number it always did. Taking it from FTS5's own tokenizer instead would
    have been a different count, and a different score."""
    from mdcx import search as B

    connection = _corpus(tmp_path, documents=4)
    identifier, text, stored = connection.execute(
        "SELECT id, text, tokens FROM passage LIMIT 1").fetchone()

    assert stored == len(B.tokenize_text(B._normalize(text))), identifier


def test_every_passage_carries_its_length(tmp_path):
    """A single missing count would send that passage down the slow path
    silently, which is the kind of thing that is only found by measuring."""
    connection = _corpus(tmp_path, documents=6)

    missing = connection.execute(
        "SELECT count(*) FROM passage WHERE tokens IS NULL").fetchone()[0]

    assert missing == 0
    assert archive._has_token_counts(connection) is True


# --- Only the terms of the question --------------------------------------------


def test_the_frequencies_asked_for_are_the_ones_the_query_uses(tmp_path):
    """`df` holds one row per term in the corpus -- 521,231 on the corpus this
    came from -- and a query uses between one and ten. Reading the table made
    every query slower as the corpus grew, for counts it discarded."""
    connection = _corpus(tmp_path)

    whole, _n, _lm = archive._corpus_statistics(connection)
    asked = archive._frequencies_for(connection, ["teorema", "espacio"])

    assert set(asked) == {"teorema", "espacio"}
    assert all(asked[t] == whole[t] for t in asked), (
        "the counts differ from the ones the whole table gives")


def test_a_term_the_corpus_does_not_hold_is_simply_absent(tmp_path):
    """The caller defaults it, as it did when the table was read whole."""
    connection = _corpus(tmp_path)

    assert archive._frequencies_for(connection, ["nothinglikethis"]) == {}


def test_more_terms_than_one_statement_holds_are_all_answered(tmp_path):
    """A query widened by the shape sieve can carry more terms than anyone
    types, and SQLite has a ceiling on host parameters."""
    connection = _corpus(tmp_path)
    many = [f"term{i:04}" for i in range(archive._TERMS_PER_STATEMENT + 40)]

    found = archive._frequencies_for(connection, many + ["teorema"])

    assert "teorema" in found


# --- The phrase branch reorders what is in hand --------------------------------


def test_the_phrase_branch_reads_only_the_documents_it_can_reorder(tmp_path):
    """It read `document.normalized_text` for the whole corpus -- 257 MB in one
    package of a 17 GB collection -- to prefer documents that were already in
    the ranking. A document it found outside the ranking could not appear in
    the reply."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")

    assert 'SELECT name, normalized_text FROM document "' in source
    assert 'f"WHERE name IN ({placeholders})", batch)' in source


def test_a_long_phrase_still_prefers_the_document_that_holds_it(tmp_path):
    """The behaviour the branch exists for, over the candidates."""
    folder = tmp_path / "md"
    folder.mkdir()
    (folder / "holds.md").write_text(
        "# Holds\n\nthe quick brown fox jumps over the lazy dog today\n",
        encoding="utf-8")
    (folder / "scattered.md").write_text(
        "# Scattered\n\nquick fox and a separate lazy dog and brown jumps\n",
        encoding="utf-8")
    archive.pack(folder, tmp_path / "p.mdcx", "k")
    connection = archive.open_package(tmp_path / "p.mdcx", "k")[0]

    found = archive.query(connection, "quick brown fox jumps over the lazy dog",
                          limit=2)

    assert found[0]["document"] == "holds"


# --- When the index is not asking about the same string ------------------------


def test_a_script_that_writes_its_vowels_as_marks_is_indexed_whole(tmp_path):
    """The defect underneath, which was not a matter of speed.

    FTS5's `unicode61` counts letters, numbers and private use as parts of a
    token, and combining marks are none of those -- so a script that writes its
    vowels as marks was cut at every one of them. Measured on one Hindi
    passage: 21 terms in the index for the 13 words it holds, and the word a
    reader would search for was not among them.

    Naming the categories fixes the index, and with it the disagreement about
    what a passage contains -- which is what a lexical score is computed from.
    """
    folder = tmp_path / "md"
    folder.mkdir()
    (folder / "hindi.md").write_text(
        "# Hindi\n\nगणित में समुच्चय सिद्धांत एक मौलिक शाखा है जो संग्रहों "
        "का अध्ययन करती है\n", encoding="utf-8")
    archive.pack(folder, tmp_path / "h.mdcx", "k")
    connection = archive.open_package(tmp_path / "h.mdcx", "k")[0]

    assert archive._index_agrees(connection) is True, (
        "the index and this module disagree about what the passage holds")
    assert archive.query(connection, "समुच्चय सिद्धांत", limit=2)


def test_the_guard_stays_for_a_package_the_index_cannot_answer_for(
        tmp_path, monkeypatch):
    """Kept as a net rather than as the fix. A package written before the
    tokenizer was declared carries its own, and this module does not own
    `unicode61`: where the two disagree the text is read instead, which is
    correct rather than fast."""
    connection = _corpus(tmp_path, documents=4)
    monkeypatch.setattr(archive, "_index_agrees", lambda c: False)

    assert archive._occurrences(connection, ["teorema"], [1, 2]) is None
    assert archive.query(connection, "teorema demostracion", limit=3)


def test_where_they_agree_the_index_is_used(tmp_path):
    """The other half: the guard must not be so cautious that nothing takes
    the fast path."""
    connection = _corpus(tmp_path, documents=4)

    assert archive._index_agrees(connection) is True
    rowids = [r[0] for r in connection.execute("SELECT id FROM passage LIMIT 5")]
    assert archive._occurrences(connection, ["teorema"], rowids) is not None
