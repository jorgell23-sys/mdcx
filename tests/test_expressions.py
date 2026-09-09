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

"""A statement and its negation were the same query.

Reported by a consumer running a corpus of mathematics, measured over 25.1
million passages:

    pq | b(b+p+q)   -- true
    pq | b(b-p-q)   -- false, one sign apart

Both retrieved the same 400 passages and the same first document, because the
rule that decides what a word is discards the symbols, and what distinguishes
those two statements is a symbol. Asked on their own the two return no terms at
all and retrieve nothing whatever.

They built a novelty check on "does the corpus already say this" and had to
take its power away again: it can say a passage is about something, not that
something was already known. That is the damage this repairs, and it is
repaired for a corpus that asks for it -- prose gains nothing from an index of
punctuation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, expressions  # noqa: E402

TRUE_ONE = "pq | b(b+p+q)"
FALSE_ONE = "pq | b(b-p-q)"


def _corpus(tmp_path, indexed=True):
    folder = tmp_path / "md"
    folder.mkdir()
    (folder / "stated.md").write_text(
        "# Theorem\n\nSe demuestra que el producto pq | b(b+p+q) para todo "
        "primo p y q que cumplan la condicion del enunciado.\n",
        encoding="utf-8")
    for i in range(12):
        (folder / f"filler{i}.md").write_text(
            f"# Filler {i}\n\nProsa sobre productos y primos que divide el "
            "enunciado, sin formula.\n", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    written = archive.pack(folder, target, "k", expressions=indexed)
    return archive.open_package(target, "k")[0], written


# --- What word matching does to a formula --------------------------------------


def test_word_matching_leaves_nothing_of_a_bare_formula():
    """The starting point, fixed here so the reason for this index is not lost:
    the two statements do not merely reduce to the same terms, they reduce to
    none."""
    from mdcx import search as B

    assert B.searchable_terms(B._normalize(TRUE_ONE)) == []
    assert B.searchable_terms(B._normalize(FALSE_ONE)) == []


def test_the_two_statements_are_told_apart(tmp_path):
    """The whole point. One sign, and they are different statements."""
    connection, written = _corpus(tmp_path)

    assert written["expressions"] >= 1
    assert archive.query(connection, TRUE_ONE, limit=3), (
        "the statement the corpus holds was not found")
    assert archive.query(connection, FALSE_ONE, limit=3) == [], (
        "the negation was answered as though the corpus stated it")


def test_the_answer_says_what_matched(tmp_path):
    """Nothing was weighted by rarity because there were no words to weigh, so
    the reply must not present the count as a lexical score."""
    connection, _written = _corpus(tmp_path)

    found = archive.query(connection, TRUE_ONE, limit=3)

    assert found[0]["matched_by"] == "expression"
    assert found[0]["terms"] == ["b(b+p+q)"]


def test_without_the_index_the_question_is_unanswerable(tmp_path):
    """What a package that did not ask for this still does -- which is nothing,
    and is why the index is offered rather than the behaviour changed."""
    connection, _written = _corpus(tmp_path, indexed=False)

    assert expressions.has_index(connection) is False
    assert archive.query(connection, TRUE_ONE, limit=3) == []


# --- What counts as an expression ----------------------------------------------


def test_prose_yields_no_expressions():
    """An index of punctuation would be worse than none: it would match
    everything and mean nothing."""
    assert expressions.extract(
        "una frase corriente, con comas y un punto.") == []


def test_a_formula_is_one_however_it_was_typed():
    """A document may carry U+2212 where a question is typed with a hyphen,
    and they are the same statement."""
    assert expressions.extract("b(b−p−q)") == expressions.extract("b(b-p-q)")


def test_a_run_of_punctuation_is_not_an_expression():
    """Something has to be operated on."""
    assert expressions.extract("--- +++ ...") == []


def test_a_flag_is_not_an_expression():
    """`--force` mixes symbols with letters and is not mathematics. It is
    admitted by the rule, which is a false positive worth knowing about rather
    than a defect: it costs an index entry and matches only itself."""
    assert expressions.extract("--force") == ["--force"]


def test_too_short_to_be_worth_an_entry():
    assert expressions.extract("a+") == []
    assert expressions.extract("f(x)") == ["f(x)"]


# --- The index says what built it ----------------------------------------------


def test_an_index_of_another_version_is_not_read_as_this_one(tmp_path,
                                                             monkeypatch):
    """The silent failure this guards: a package written under a later rule,
    read under this one, answers about a different spelling."""
    connection, _written = _corpus(tmp_path)
    assert expressions.has_index(connection) is True

    monkeypatch.setattr(expressions, "EXPRESSION_VERSION",
                        expressions.EXPRESSION_VERSION + 1)
    assert expressions.has_index(connection) is False


def test_the_lookup_is_exact(tmp_path):
    """Anything that treats two statements as near enough returns the defect
    this was built to remove."""
    connection, _written = _corpus(tmp_path)

    found = expressions.passages_stating(connection,
                                         ["b(b+p+q)", "b(b-p-q)"])

    assert "b(b+p+q)" in found
    assert "b(b-p-q)" not in found
