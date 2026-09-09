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

"""Finding a quotation that does not respect the cut between passages.

A quoted phrase often starts in one passage and ends in the next. The package
has always answered that by keeping `normalized_text` -- the whole text of every
document -- and reading it. That has no length limit and costs a second copy of
the corpus: measured by a consumer on one package, 257.4 MB beside the 257.3 MB
of the passages themselves, a third of the file.

The other shape indexes the join instead: the tail of each passage against the
head of the next, in a contentless FTS5 table that keeps the index and not the
text. It finds a quotation spanning one cut, and not one longer than the join --
so it is a choice between two costs, not a replacement.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402

TAIL = "el teorema fundamental establece que toda funcion continua sobre un"
HEAD = "intervalo cerrado alcanza su maximo y su minimo en dicho intervalo"
QUOTE = TAIL + " " + HEAD


def _corpus(tmp_path):
    random.seed(53)
    folder = tmp_path / "md"
    folder.mkdir()
    for i in range(12):
        blocks = [f"Relleno {j} del documento {i}, con prosa que no dice nada."
                  for j in range(5)]
        if i == 7:
            blocks.insert(2, TAIL)
            blocks.insert(3, HEAD)
        (folder / f"d{i:02}.md").write_text(
            f"# Doc {i}\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return folder


def _packed(tmp_path, quotes):
    folder = _corpus(tmp_path)
    target = tmp_path / f"{quotes}.mdcx"
    written = archive.pack(folder, target, "k", quotes=quotes)
    return archive.open_package(target, "k")[0], written


# --- Both shapes find it -------------------------------------------------------


def test_a_quotation_across_a_cut_is_found_by_the_whole_text(tmp_path):
    """What the second copy is for, and it still works."""
    connection, _written = _packed(tmp_path, "document")

    found = archive.query(connection, QUOTE, limit=3)

    assert found and found[0]["document"] == "d07"


def test_a_quotation_across_a_cut_is_found_by_the_indexed_join(tmp_path):
    """The same answer without the copy. The quotation is in no single passage
    -- that is the whole difficulty -- and the join holds it whole."""
    connection, written = _packed(tmp_path, "boundary")

    assert written["quotes"] == "boundary"
    assert written["passage_edges"] > 0
    found = archive.query(connection, QUOTE, limit=3)

    assert found and found[0]["document"] == "d07"


def test_the_copy_is_not_written_when_the_joins_are_indexed(tmp_path):
    """Otherwise this would cost more than it saves, which is the whole
    point of offering it."""
    connection, _written = _packed(tmp_path, "boundary")

    held = connection.execute(
        "SELECT count(*) FROM document WHERE normalized_text IS NOT NULL"
    ).fetchone()[0]

    assert held == 0
    assert archive._has_edges(connection) is True


def test_the_copy_is_written_by_default(tmp_path):
    """The default is the shape without a length limit; nothing changes for a
    caller who does not ask."""
    connection, written = _packed(tmp_path, "document")

    held = connection.execute(
        "SELECT count(*) FROM document WHERE normalized_text IS NOT NULL"
    ).fetchone()[0]

    assert held == 12
    assert written["quotes"] == "document"
    assert archive._has_edges(connection) is False


# --- What comes back is a passage of the corpus --------------------------------


def test_the_joined_text_is_never_returned_as_a_passage(tmp_path):
    """The join is text that exists nowhere: it is the end of one passage
    glued to the start of another. A match gives back the rowid, which is the
    passage that opens the join and does exist."""
    connection, _written = _packed(tmp_path, "boundary")

    found = archive.query(connection, QUOTE, limit=3)

    for result in found:
        stored = connection.execute(
            "SELECT count(*) FROM passage WHERE text = ?",
            (result["passage"],)).fetchone()[0]
        assert stored == 1, "a passage was returned that is not in the corpus"


# --- The header says which shape it is -----------------------------------------


def test_the_header_says_how_quotations_are_answered(tmp_path):
    """A reader comparing two packages needs to know which of the two shapes
    they hold, and the header is read without the key."""
    folder = _corpus(tmp_path)
    archive.pack(folder, tmp_path / "b.mdcx", "k", quotes="boundary")

    _connection, header = archive.open_package(tmp_path / "b.mdcx", "k")

    assert header["quotes"] == "boundary"


def test_a_shape_that_does_not_exist_is_refused(tmp_path):
    """Before the corpus is indexed, since indexing is the expensive half."""
    folder = _corpus(tmp_path)

    try:
        archive.pack(folder, tmp_path / "x.mdcx", "k", quotes="whole-corpus")
    except ValueError as refused:
        assert "document" in str(refused) and "boundary" in str(refused)
    else:
        raise AssertionError("an unknown strategy was accepted")


# --- What it cannot do, stated -------------------------------------------------


def test_a_quotation_longer_than_the_join_is_the_limit_of_this_shape(tmp_path):
    """Said out loud rather than discovered. The join carries a bounded amount
    of each side, so a quotation longer than it crosses out of the window --
    which the whole copy would still find. That is the trade being offered."""
    assert archive.EDGE_CHARACTERS > 0
    # About forty words a side, so some eighty words spanning one cut fit.
    assert 100 <= archive.EDGE_CHARACTERS <= 1000
