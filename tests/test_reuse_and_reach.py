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

"""What a package hands to the next one, and what it can be asked about itself.

Three things a corpus used as a living memory needs and did not have:

Its calibration had to survive being rebuilt. Adding one document with `--reuse`
recovered the expensive half, the vectors, and dropped the cheap half, the
questions -- so the threshold silently reverted to being estimated from
passages. The stored number barely moves; the margin applied to it goes from
0.95 to 0.60.

Whether a package is about a question had to be answerable per package. It was
computed only across all of them at once, which is the right question for a
warning and the wrong one for deciding which of several packages answers.

And writing often had to be affordable. Compressing is a property of the whole
file, so it costs the same whatever was added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import semantic as S  # noqa: E402

pytestmark = pytest.mark.skipif(
    not S.available(),
    reason="the reach is a cosine, and without the extra there are no vectors")

FOCUS = ["como conviene regar para gastar menos agua",
         "que hacer contra la helada tardia",
         "cuando se poda un duraznero"]


@pytest.fixture
def orchard(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in (
            ("riego", "El riego por goteo ahorra agua frente al riego por surco."),
            ("helada", "Contra la helada tardia conviene regar por aspersion."),
            ("poda", "El duraznero se poda en invierno, tras la caida de la hoja.")):
        (folder / f"{name}.md").write_text(f"# {name}\n\n{text}\n", encoding="utf-8")
    return folder


# --- The calibration survives the next packing -------------------------------


def test_reusing_a_package_keeps_the_questions_it_was_calibrated_with(
        orchard, tmp_path):
    """The defect: the cheap half was dropped and the expensive half kept.

    What makes it hard to notice is that the stored number barely moves. The
    signal is `answerable_at_from`, and one has to know to look at it.
    """
    first = archive.pack(orchard, tmp_path / "A.mdcx", "k",
                         semantic=True, focus=FOCUS)
    assert first["answerable_at_from"] == "focus", "the fixture was not calibrated"

    (orchard / "abono.md").write_text("# abono\n\nEl abono verde aporta materia.\n",
                                      encoding="utf-8")
    second = archive.pack(orchard, tmp_path / "B.mdcx", "k", semantic=True,
                          reuse_from=tmp_path / "A.mdcx")

    assert second["answerable_at_from"] == "focus"
    assert second["focus_inherited"] is True, "inherited silently is the same fault"


def test_questions_given_on_the_spot_still_win(orchard, tmp_path):
    """Inheriting must not take recalibration away.

    It acts only where nothing was passed, so packing again with different
    questions works exactly as it did.
    """
    archive.pack(orchard, tmp_path / "A.mdcx", "k", semantic=True, focus=FOCUS)

    again = archive.pack(orchard, tmp_path / "C.mdcx", "k", semantic=True,
                         reuse_from=tmp_path / "A.mdcx",
                         focus=["cuanto nitrogeno pide el maiz"])

    assert not again.get("focus_inherited")
    assert again["answerable_at_from"] == "focus"


def test_a_package_calibrated_from_passages_hands_on_nothing(orchard, tmp_path):
    """There is nothing to inherit, and claiming otherwise would be a lie."""
    archive.pack(orchard, tmp_path / "P.mdcx", "k", semantic=True)

    after = archive.pack(orchard, tmp_path / "Q.mdcx", "k", semantic=True,
                         reuse_from=tmp_path / "P.mdcx")

    assert not after.get("focus_inherited")
    # Either estimated from passages or not measured at all -- a corpus this
    # small gives the probes too little to work with. What must not happen is
    # `focus`, which would mean questions appeared from nowhere.
    assert after.get("answerable_at_from") != "focus"


# --- Asking one package whether it is about something ------------------------


def test_a_package_can_be_asked_how_near_it_comes(orchard, tmp_path):
    """The number existed and was only computed over every package at once."""
    archive.pack(orchard, tmp_path / "A.mdcx", "k", semantic=True, focus=FOCUS)
    connection, _ = archive.open_package(tmp_path / "A.mdcx", "k")

    measured = archive.closeness(connection, "cuando conviene podar el duraznero")

    assert measured is not None
    near, clear = measured
    assert 0.0 < near <= 1.0
    assert clear >= 0.0


def test_a_package_without_vectors_offers_no_number_rather_than_a_bad_one(
        orchard, tmp_path):
    """Inventing one from BM25 would be the mistake this replaced: a BM25 score
    depends on the corpus it was measured in, so two packages cannot be
    compared by it."""
    archive.pack(orchard, tmp_path / "W.mdcx", "k")
    connection, _ = archive.open_package(tmp_path / "W.mdcx", "k")

    assert archive.closeness(connection, "cualquier cosa") is None
    assert archive.answers(connection, "cualquier cosa") is False


def test_the_decision_uses_the_margin_that_matches_the_calibration(
        orchard, tmp_path):
    """The part that is easiest to get wrong, and getting it wrong raises no
    error: it produces a decider that lets everything through.

    A threshold taken from the questions themselves is already the threshold.
    Scaling it by the passage share again would put the cut a third lower.
    """
    archive.pack(orchard, tmp_path / "A.mdcx", "k", semantic=True, focus=FOCUS)
    connection, _ = archive.open_package(tmp_path / "A.mdcx", "k")
    reach = archive.answerable_at(connection)

    assert archive.calibrated_from_questions(connection)
    # Just under what the questions reached: refused with the right margin,
    # admitted with the wrong one.
    just_under = reach * 0.90
    assert archive._below_reach(just_under, 0.0, reach, from_questions=True)
    assert not archive._below_reach(just_under, 0.0, reach, from_questions=False)


def test_the_rule_lives_in_one_place(orchard, tmp_path):
    """The server used to hold it alone, so anything else that wanted to branch
    on it had to rebuild it out of private functions."""
    from mdcx import mcp_server

    assert mcp_server.NOTHING_NEAR is archive.NOTHING_NEAR
    assert mcp_server.ANSWERS_AT_SHARE is archive.ANSWERS_AT_SHARE
    assert mcp_server.ASKED_MARGIN is archive.ASKED_MARGIN
    assert mcp_server.TAIL_AT is archive.TAIL_AT


def test_two_packages_are_told_apart_where_one_aggregate_could_not(tmp_path):
    """The case the report was written from: two cells with different roles.

    Confusing them would let a conclusion the system reached itself be cited
    afterwards as established knowledge.
    """
    orchard = tmp_path / "huerta"
    orchard.mkdir()
    (orchard / "poda.md").write_text(
        "# poda\n\nEl duraznero se poda en invierno, tras la caida de la hoja.\n",
        encoding="utf-8")
    algebra = tmp_path / "algebra"
    algebra.mkdir()
    (algebra / "regla.md").write_text(
        "# regla\n\nLa derivada de una funcion compuesta se obtiene por la "
        "regla de la cadena.\n", encoding="utf-8")

    archive.pack(orchard, tmp_path / "H.mdcx", "k", semantic=True)
    archive.pack(algebra, tmp_path / "M.mdcx", "k", semantic=True)
    huerta, _ = archive.open_package(tmp_path / "H.mdcx", "k")
    mate, _ = archive.open_package(tmp_path / "M.mdcx", "k")

    question = "como se calcula la derivada de una funcion compuesta"
    assert (archive.closeness(mate, question)[0]
            > archive.closeness(huerta, question)[0]), (
        "the package that answers must come out nearer")


# --- Writing often ------------------------------------------------------------


def test_fast_trades_size_for_time_and_the_package_still_opens(orchard, tmp_path):
    """A cheaper compression, not a different format."""
    ordinary = archive.pack(orchard, tmp_path / "N.mdcx", "k")
    quick = archive.pack(orchard, tmp_path / "F.mdcx", "k", fast=True)

    assert quick["bytes_compressed"] >= ordinary["bytes_compressed"]

    connection, _ = archive.open_package(tmp_path / "F.mdcx", "k")
    assert archive.query(connection, "duraznero", limit=2), "packed but unreadable"


def test_the_fixed_price_of_a_write_is_reported(orchard, tmp_path):
    """What lets a caller decide how often to write.

    Compressing and encrypting are properties of the whole file, so they cost
    the same whether one document was added or the corpus was rebuilt.
    """
    written = archive.pack(orchard, tmp_path / "N.mdcx", "k")

    assert "seconds_compress" in written
    assert "seconds_encrypt" in written


# --- The one that was not reported -------------------------------------------


def test_a_calibrated_package_can_actually_be_served(orchard, tmp_path,
                                                     monkeypatch):
    """Found while building the two above, and worse than either.

    `--focus` has shipped since 1.12.0, and the MCP `search` tool failed
    outright against any package that used it: the reply never arrived, the
    tool raised. Two causes in one line -- the value was read back through
    `json.loads`, which raises on the bare word `focus` that meta actually
    stores, and the name `json` was never imported in that module at all.

    Nobody hit it because a package that was never given questions has no such
    row, and the check short-circuits before reaching the line.
    """
    import asyncio
    import importlib
    import json as _json

    archive.pack(orchard, tmp_path / "A.mdcx", "k", semantic=True, focus=FOCUS)
    monkeypatch.setenv("MDCX_FILE", str(tmp_path / "A.mdcx"))
    monkeypatch.setenv("MDCX_KEY", "k")

    from mdcx import mcp_server
    importlib.reload(mcp_server)

    assert mcp_server._calibrated_from_questions(), (
        "the calibration was not recognised, so the wrong margin would apply")

    output = asyncio.run(mcp_server.create_server().call_tool(
        "search", {"query": "cuando se poda un duraznero"}))
    assert not output.is_error, output.content
    assert _json.loads(output.content[0].text)["found"] > 0
