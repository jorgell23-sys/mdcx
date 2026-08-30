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

"""A package holds two signals about a question, and they disagree.

The cosine says how near the corpus comes. It cannot tell the senses of a
homonym apart, because a multilingual embedding places them together: a package
of algebra accepted "graph coloring adjacent vertices different colors" at
0.6553 against a threshold of 0.5661 and returned lessons on comparing graphs
and on the ellipse. The word the question turns on, `coloring`, it had never
seen.

The vocabulary says which words are new. It is literal, so across languages it
returns every word of the question and measures which language the corpus is
written in rather than what it knows.

Each is wrong where the other is right, and until now neither said so.

And a third thing, small and sharp: a cosine was coming back above 1. Vectors
are stored in half precision, and rounding costs the normalisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import semantic as S  # noqa: E402

needs_vectors = pytest.mark.skipif(
    not S.available(), reason="both signals need the meaning index")


@pytest.fixture
def algebra(tmp_path):
    """English, and about graphs in the sense of plots.

    Wide enough that an English question shares most of its words with it.
    A corpus of three sentences has not seen `you` or `different` either, and
    then every question looks foreign -- which is a real property of a small
    corpus and not what these tests are about.
    """
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in (
            ("slope", "The slope of a line is the ratio of rise to run, and "
                      "you can read it from the graph of any linear function. "
                      "Two different lines may share the same slope when they "
                      "are parallel to each other."),
            ("factor", "To factor a quadratic polynomial you find two numbers "
                       "whose product is the constant term and whose sum is "
                       "the middle coefficient. Different polynomials factor "
                       "in different ways, and some do not factor at all over "
                       "the integers."),
            ("ellipse", "The ellipse is a conic section, and its graph is "
                        "centred at the origin of the plane. Every point on it "
                        "keeps the same total distance from the two foci, "
                        "which is what makes drawing one with a string work."),
            # `color` is here on purpose: it is a cognate, and a cognate is
            # what switched the language flag off.
            ("colour", "A graph may be drawn in any color, and the colors "
                       "chosen for adjacent regions of a chart should be "
                       "different enough to tell apart. This is a matter of "
                       "presentation and not of the mathematics."),
            ("lines", "Comparing graphs of several functions on one pair of "
                      "axes shows where they meet. The point where two lines "
                      "cross is the solution of the system they describe.")):
        (folder / f"{name}.md").write_text(f"# {name}\n\n{text}\n", encoding="utf-8")
    return folder


# --- A cosine stays inside the range a cosine has -----------------------------


@needs_vectors
def test_a_passage_against_itself_does_not_exceed_one(algebra, tmp_path):
    """Four separately built packages all topped out at exactly 1.000428, which
    is what says it is the storage format and not the text.

    Nothing decided on it changes -- thresholds sit around 0.56 and the excess
    is 4e-4. What broke is the invariant, and an invariant that almost holds is
    worse than one that does not: it sent a reader looking for an explanation,
    who found a plausible wrong one and stopped.
    """
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    for (passage,) in connection.execute("SELECT text FROM passage"):
        near, _ = archive.closeness(connection, passage)
        assert near <= 1.0, f"a cosine of {near} against the passage itself"
        assert near >= -1.0


@needs_vectors
def test_the_stored_vectors_are_unit_length_when_read(algebra, tmp_path):
    """Fixed where it happens rather than clamped where it shows.

    Clamping the result would hide it; the vectors themselves were no longer
    unit length, so everything computed from them was slightly not a cosine.
    """
    import numpy as np

    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    _, matrix = archive._vectors(connection)
    norms = np.linalg.norm(matrix, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"norms range {norms.min():.6f} to {norms.max():.6f}")


# --- The two signals disagree, and the reply says both ------------------------


@needs_vectors
def test_a_question_whose_defining_word_is_absent_is_flagged(algebra, tmp_path):
    """The homonym: `graph` as a plot, `graph` as a graph.

    The verdict is not overruled -- whether an unfamiliar word should refuse a
    query depends on what the query is for -- but it no longer travels alone.
    """
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    judged = archive.assess(
        connection, "graph coloring adjacent vertices different colors")

    assert "coloring" in judged["unknown_terms"]
    assert judged["unknown_terms_meaningful"] is True
    assert judged["closeness"] is not None


@needs_vectors
def test_a_question_the_corpus_answers_brings_no_strange_word(algebra, tmp_path):
    """Otherwise the signal would refuse everything and mean nothing."""
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    judged = archive.assess(connection, "the slope of a line and its graph")

    assert judged["answers"] is True
    assert judged["unknown_terms"] == []


@needs_vectors
def test_the_verdict_is_reported_and_not_overruled(algebra, tmp_path):
    """mdcx does not decide this. An unknown word can be peripheral -- "the
    slope of a line drawn in Patagonia" is answerable and `patagonia` is
    unknown -- and which it is depends on what the question is for."""
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    question = "the slope of a line drawn in Patagonia"
    judged = archive.assess(connection, question)

    assert "patagonia" in judged["unknown_terms"]
    assert judged["answers"] == archive.answers(connection, question), (
        "the vocabulary changed the verdict instead of accompanying it")


# --- Across languages the signal saturates, and says so -----------------------


@needs_vectors
def test_a_question_in_another_language_is_not_reported_as_unknown(
        algebra, tmp_path):
    """Every term comes back unknown, which measures the corpus's language.

    This is the promoted way to use mdcx -- ask in your own language against a
    corpus in another -- so it is exactly where a consumer would adopt the
    signal and get one that is always positive.
    """
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    spanish = archive.unfamiliar(
        connection, "como se factoriza un polinomio de segundo grado")

    assert spanish["share"] == 1.0, "premise: every term looks unknown"
    assert spanish["cross_language"] is True
    assert archive.assess(
        connection,
        "como se factoriza un polinomio de segundo grado"
    )["unknown_terms_meaningful"] is False


@needs_vectors
def test_the_share_is_reported_because_one_is_the_signature(algebra, tmp_path):
    """A consumer that cannot detect the language can still see 1.00."""
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    strange = archive.unfamiliar(connection, "the slope of a line and its graph")

    assert 0.0 <= strange["share"] < 1.0
    assert strange["considered"] > 0


def test_a_text_with_nothing_to_consider_does_not_claim_familiarity(tmp_path):
    """Zero terms is not "nothing unknown"; it is a question the data cannot
    answer, and a share of 0.0 would be the wrong answer to it."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# a\n\nThe slope of a line.\n", encoding="utf-8")
    archive.pack(folder, tmp_path / "M.mdcx", "k")
    connection, _ = archive.open_package(tmp_path / "M.mdcx", "k")

    strange = archive.unfamiliar(connection, "a b c")

    assert strange["considered"] == 0
    assert strange["share"] is None


# --- Through the MCP ----------------------------------------------------------


@needs_vectors
def test_the_mcp_reply_carries_the_strange_words(algebra, tmp_path, monkeypatch):
    """Where a consumer needs it without having to know to ask."""
    import asyncio
    import importlib
    import json

    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    monkeypatch.setenv("MDCX_FILE", str(target))
    monkeypatch.setenv("MDCX_KEY", "k")

    from mdcx import mcp_server
    importlib.reload(mcp_server)
    server = mcp_server.create_server()

    def reply(arguments: dict) -> dict:
        output = asyncio.run(server.call_tool("search", arguments))
        assert not output.is_error, output.content
        return json.loads(output.content[0].text)

    # Against the package, not pooled. Intersecting across packages emptied the
    # signal as the library grew: four packages each lacked something different
    # and no term was missing from all four, while three of them had never seen
    # the word the question turned on. And what a reader needs is not whether
    # some package knows the word -- it is whether the one the passage they are
    # about to cite came from knows it.
    flagged = reply({"query": "graph coloring adjacent vertices"})
    assert "coloring" in flagged["unknown_terms"]["M.mdcx"]

    # And silent where the words are all familiar, so the field means something
    # when it is there.
    plain = reply({"query": "the slope of a line and its graph"})
    assert "unknown_terms" not in plain

    # And silent across languages, where every word would be listed.
    crossed = reply({"query": "como se factoriza un polinomio de segundo grado"})
    assert "unknown_terms" not in crossed


# --- What using 1.18.0 turned up ---------------------------------------------


@needs_vectors
def test_both_doors_to_the_same_number_agree(algebra, tmp_path):
    """The renormalisation reached one caller and not the other.

    `closeness` came back at exactly 1.000000 while `cosines` -- the one the MCP
    builds its reply from -- still returned 1.000448 for the same text on the
    same package in the same process. It looked like a server serving the old
    version, and it was a repair that had not covered all its doors.
    """
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    for (passage,) in connection.execute("SELECT text FROM passage"):
        near, _ = archive.closeness(connection, passage)
        assert max(archive.cosines(connection, passage)) <= 1.0
        assert abs(max(archive.cosines(connection, passage)) - near) < 1e-6


def test_the_query_vector_is_unit_length(algebra):
    """Where the remaining excess actually came from.

    The stored side had been repaired and the excess stayed, because the model
    normalises in half precision on the accelerator: what `encode` returns has
    norm 1 + 4.5e-4, exactly the excess that was observed, and casting to
    float32 preserves it rather than removing it.
    """
    if not S.available():
        pytest.skip("no encoder")
    import numpy as np

    vectors = S.encode(["# Preliminares", "a longer sentence about slopes"],
                       role="query")

    norms = np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"norms {norms}"


def test_the_ordering_survived_the_repair(algebra, tmp_path):
    """`closeness` reads the first element as the best and one further down as
    the tail, so a change here that dropped the sort would be silent."""
    if not S.available():
        pytest.skip("no encoder")
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k", semantic=True)
    connection, _ = archive.open_package(target, "k")

    scores = archive.cosines(connection, "the slope of a line")

    assert scores == sorted(scores, reverse=True)


def test_one_shared_word_does_not_switch_off_the_language_flag(
        algebra, tmp_path):
    """A cognate was enough, and between related languages there is always one.

    `color`, `radio`, `natural`, `total`, `error`, `region`, a number, an
    initialism or a proper noun all did it. Worse than losing the flag: with it
    off, the reply said the lexical signal was readable over a share of 0.93
    that was pure language.
    """
    target = tmp_path / "M.mdcx"
    archive.pack(algebra, target, "k")
    connection, _ = archive.open_package(target, "k")

    base = "como se factoriza un polinomio de segundo grado"
    plain = archive.unfamiliar(connection, base)
    with_cognate = archive.unfamiliar(connection, base + " de color")

    assert plain["cross_language"] is True
    assert with_cognate["share"] < 1.0, "premise: the cognate lowered the share"
    assert with_cognate["cross_language"] is True
    assert with_cognate["meaningful"] is False, (
        "a share this high was reported as a readable signal")


def test_the_cut_is_not_an_equality(algebra, tmp_path):
    """The shape of the mistake, kept so it is not made again.

    An exact cut on a quantity that spreads is the wrong shape -- the same
    finding NOTHING_NEAR and STANDS_CLEAR reached from the other side.
    """
    assert 0.17 < archive.CROSS_LANGUAGE_SHARE <= 0.7143, (
        "the cut must mark every measured crossing and no same-language question")


@needs_vectors
def test_the_reach_of_each_package_is_published(algebra, tmp_path, monkeypatch):
    """Two numbers of different provenance were read as describing one package.

    `similarity` is over every package pooled; `answerable_at` is the lowest of
    their reaches, so the threshold in force comes from the narrowest rather
    than from the one a passage came from.
    """
    import asyncio
    import importlib
    import json

    questions = ["the slope of a line and its graph",
                 "how to factor a quadratic polynomial"]
    first = tmp_path / "A.mdcx"
    second = tmp_path / "B.mdcx"
    archive.pack(algebra, first, "k", semantic=True, focus=questions)
    archive.pack(algebra, second, "k", semantic=True, focus=questions[:1])

    monkeypatch.setenv("MDCX_FILE", f"{first}{__import__('os').pathsep}{second}")
    monkeypatch.setenv("MDCX_KEY", "k")
    from mdcx import mcp_server
    importlib.reload(mcp_server)

    output = asyncio.run(mcp_server.create_server().call_tool(
        "search", {"query": "the slope of a line"}))
    assert not output.is_error, output.content
    answer = json.loads(output.content[0].text)

    each = answer.get("answerable_at_by_package")
    assert each and len(each) == 2, "the aggregate travelled without its parts"
    assert answer["answerable_at"] == round(min(each.values()), 4)
