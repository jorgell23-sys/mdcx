"""Checks the restriction of a search to one side of a correspondence.

A collection of correspondence has a direction, and a question is usually about
one side of it. The direction is taken from the top-level folder and stored with
each document, and a query can be restricted to it.

The restriction was written twice and the two halves did not agree on the form
of the value. The stored form is uppercase; the callers all produce lowercase --
the CLI declares its choices that way and the server lowercases what it is
given. The lexical engine folded the case before comparing and the dense one did
not, so a restricted query returned its lexical results and silently dropped
every dense one. Where the query was written in the language of the documents
this looked like nothing at all, because the lexical engine had already found
them; where it was not, the reply came back empty and read as a corpus holding
nothing on the subject.

That is the failure these tests hold shut: not that the filter works, but that
both engines agree on what they are filtering by.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, semantic  # noqa: E402

needs_model = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")


def _build(root: Path, *, with_semantics: bool) -> Path:
    """A collection with both directions present, in English."""
    received = root / "src" / "Received"
    issued = root / "src" / "Sent"
    received.mkdir(parents=True)
    issued.mkdir(parents=True)
    (received / "incoming.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Photosynthesis\n\n"
        "Plants turn sunlight into sugar using the chlorophyll in their "
        "leaves, and release oxygen while doing so.\n", encoding="utf-8")
    (issued / "outgoing.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Printing\n\n"
        "Gutenberg built the printing press with movable type in Mainz, "
        "and the page was set one letter at a time.\n", encoding="utf-8")
    package = root / "corpus.mdcx"
    archive.pack(root / "src", package, "k", semantic=with_semantics)
    return package


@pytest.fixture
def lexicon(tmp_path):
    connection, _ = archive.open_package(_build(tmp_path, with_semantics=False), "k")
    return connection


@pytest.fixture
def with_semantics(tmp_path):
    connection, _ = archive.open_package(_build(tmp_path, with_semantics=True), "k")
    return connection


def test_the_direction_comes_from_the_top_level_folder(lexicon):
    stored = {row[0] for row in
                   lexicon.execute("SELECT DISTINCT source FROM document")}
    assert stored == {"RECEIVED", "SENT"}


def test_a_restriction_returns_only_that_side(lexicon):
    received = archive.query(lexicon, "photosynthesis sunlight", only="received")
    issued = archive.query(lexicon, "printing press movable type", only="sent")
    assert received and all("incoming" in r["document"] for r in received)
    assert issued and all("outgoing" in r["document"] for r in issued)


def test_a_restriction_excludes_the_other_side(lexicon):
    """The document exists and holds the words; the restriction is what removes it."""
    assert archive.query(lexicon, "printing press movable type")
    assert not archive.query(lexicon, "printing press movable type", only="received")


def test_the_case_a_caller_writes_does_not_change_the_answer(lexicon):
    """Every caller writes lowercase and the column holds uppercase."""
    expected = archive.query(lexicon, "photosynthesis sunlight", only="RECEIVED")
    assert expected
    for written in ("received", "Received", "rEcEiVeD"):
        assert [r["document"] for r in
                archive.query(lexicon, "photosynthesis sunlight", only=written)] == \
               [r["document"] for r in expected], f"{written!r} answered differently"


def test_omitting_the_direction_reaches_both(lexicon):
    documents = {r["document"] for r in
                  archive.query(lexicon, "photosynthesis printing", limit=8)}
    assert len(documents) == 2


@needs_model
def test_a_restricted_query_still_crosses_languages(with_semantics):
    """The regression: the dense engine is the only one that can answer here.

    The query is Spanish and the documents are English, so they share no term
    and the lexical engine contributes nothing. If the restriction drops the
    dense results, what comes back is empty -- which is what happened, and read
    as a corpus with nothing on the subject rather than as a filter fault.
    """
    query = "como convierten las plantas la luz solar en azucar"
    assert not archive.query(with_semantics, query, mode="lexical"), (
        "the premise of this test is that word matching cannot answer it")

    without_restriction = archive.query(with_semantics, query, limit=3)
    assert without_restriction, "the dense engine should answer this on its own"

    restricted = archive.query(with_semantics, query, limit=3, only="received")
    assert restricted, (
        "a restriction to the side that holds the answer returned nothing")
    assert all("incoming" in r["document"] for r in restricted)


@needs_model
def test_a_restriction_still_excludes_the_other_side_by_meaning(with_semantics):
    """Reaching across languages must not reach past the restriction."""
    restricted = archive.query(
        with_semantics, "quien invento la imprenta de tipos moviles",
        limit=3, only="received")
    assert all("outgoing" not in r["document"] for r in restricted)


def test_the_values_the_cli_offers_are_the_values_that_match(lexicon):
    """The choices a user is given have to be choices the column can answer.

    This is the shape of the original fault: two places agreeing on the concept
    and not on the string. The parser is built inside main and cannot be
    inspected without running it, so the declaration is read from the source --
    which is also what a reviewer would compare by eye, and does not go stale
    when the parser is rearranged.
    """
    import ast

    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")
    declared: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--only"):
            continue
        for word in node.keywords:
            if word.arg == "choices":
                declared = set(ast.literal_eval(word.value))
    assert declared, "the --only argument no longer declares its choices"

    stored = {row[0] for row in
                   lexicon.execute("SELECT DISTINCT source FROM document")}
    for choice in declared:
        assert choice.upper() in stored, (
            f"the CLI offers {choice!r}, which no document is stored as")
