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

"""A preference that could not be applied says so.

`--prefer recent` orders the fusion of two engines, so there has to be a fusion,
and it orders by date, so something in the answer has to carry one. Where either
is missing the preference is accepted, has no effect, and -- until this -- said
nothing. That is not a wrong answer; it is an answer indistinguishable from one
where the preference applied and found nothing to move.

The case that misleads most is not the old package. It is a package that *has*
dates -- `info` prints `Dated: 3 of 3` -- and lacks the meaning index. Everything
the caller can see says the preference was applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402


@pytest.fixture
def dated(tmp_path):
    """Three works with dates, packed without a meaning index."""
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, date in (("old", "1998-05-10"), ("mid", "2010-03-01"),
                       ("new", "2024-11-20")):
        (folder / f"{name}.md").write_text(
            f"---\ntitle: {name}\ndated: {date}\n---\n\n"
            "Drip irrigation scheduling and lateral pressure loss.\n",
            encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(folder, target, "k")
    connection, header = archive.open_package(target, "k")
    return connection, header


def test_the_package_has_dates_and_the_preference_still_cannot_apply(dated):
    """The report's own reproduction, and the one that misleads.

    `info` announces the dates, `--prefer recent` is accepted, and the order is
    the one it would have been anyway.
    """
    connection, header = dated
    assert header["dated_documents"] == [3, 3], "the fixture lost its dates"

    notes: dict = {}
    found = archive.query(connection, "drip irrigation", limit=3,
                          prefer="recent", notes=notes)

    assert found, "the answer itself must not change"
    assert notes["prefer_applied"] is False
    assert "meaning index" in notes["prefer_reason"]


def test_nothing_is_said_when_no_preference_was_asked_for(dated):
    """A caller that never mentioned `prefer` should not read about it."""
    connection, _ = dated

    notes: dict = {}
    archive.query(connection, "drip irrigation", limit=3, notes=notes)

    assert notes == {}


def test_asking_for_one_engine_alone_is_its_own_reason(dated):
    """`lexical` and `semantic` are not the same case as a package that cannot.

    Telling the caller "no meaning index" when they asked for lexical mode would
    send them looking for a fault in the package.
    """
    connection, _ = dated

    notes: dict = {}
    archive.query(connection, "drip irrigation", limit=3, mode="lexical",
                  prefer="recent", notes=notes)

    assert notes["prefer_applied"] is False
    assert "words alone" in notes["prefer_reason"]


def test_a_package_that_records_no_dates_says_that_and_not_something_else(
        tmp_path):
    """What the packages built before 1.15.0 reach.

    The reason has to name the missing dates rather than the missing index,
    because the two are fixed differently: one by repacking with `--dates`, the
    other by packing with `--multilingual`.
    """
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\nDrip irrigation.\n", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(folder, target, "k")
    connection, _ = archive.open_package(target, "k")

    # Asked of the reason directly: reaching this branch through `query` needs
    # the multilingual extra, and the reason has to be right either way.
    assert "pack it again with --dates" in archive._why_no_dates(connection)


def test_the_three_ways_of_having_no_date_are_told_apart(tmp_path):
    """Each is undone differently, so one sentence for all three misleads.

    Answering "nothing in this answer carries a date" to a package that holds
    no dates at all sends someone hunting through their query.
    """
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "bare.md").write_text("# Bare\n\nDrip irrigation.\n", encoding="utf-8")

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "bare.md").write_text("# Bare\n\nDrip irrigation.\n", encoding="utf-8")
    (mixed / "dated.md").write_text(
        "---\ntitle: D\ndated: 2024-01-01\n---\n\nLateral pressure loss.\n",
        encoding="utf-8")

    # Has the columns, has nothing in them: packed without --dates.
    empty = tmp_path / "empty.mdcx"
    archive.pack(bare, empty, "k")
    connection, _ = archive.open_package(empty, "k")
    assert "pack it again" in archive._why_no_dates(connection)

    # Dated, and the gap is in the answer rather than in the package.
    some = tmp_path / "some.mdcx"
    archive.pack(mixed, some, "k")
    connection, _ = archive.open_package(some, "k")
    assert "though the package has some" in archive._why_no_dates(connection)

    # A package from before the columns existed.
    old = tmp_path / "old.mdcx"
    archive.pack(bare, old, "k")
    connection, _ = archive.open_package(old, "k")
    connection.execute("ALTER TABLE document DROP COLUMN dated")
    assert "written before mdcx" in archive._why_no_dates(connection)


# --- What the caller is told -------------------------------------------------


def test_the_reason_names_something_the_caller_can_act_on(dated):
    """A reason that only restates the outcome is no better than silence."""
    connection, _ = dated

    notes: dict = {}
    archive.query(connection, "drip irrigation", limit=3, prefer="recent",
                  notes=notes)

    reason = notes["prefer_reason"]
    assert not reason.startswith("prefer"), "the reason restates the field name"
    assert len(reason.split()) >= 5, "too terse to act on"


def test_the_command_line_prints_the_reason_and_only_when_there_is_one(
        tmp_path, capsys, monkeypatch):
    """Said when it failed, silent when it worked.

    Printing a line either way would make the line meaningless: an unchanged
    order with nothing under it is how the caller learns the preference ran and
    found nothing to move.
    """
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("---\ntitle: A\ndated: 2024-01-01\n---\n\n"
                                 "Drip irrigation scheduling.\n", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(folder, target, "k")
    monkeypatch.setenv("MDCX_KEY", "k")

    def run(*extra: str) -> str:
        monkeypatch.setattr(
            sys, "argv",
            ["mdcx", "search", str(target), "drip irrigation", *extra])
        archive.main()
        return capsys.readouterr().out

    assert "was NOT applied" in run("--prefer", "recent")
    assert "NOT applied" not in run(), "spoke about a preference nobody asked for"


# --- Through the MCP, which is where the report saw it ------------------------


def test_the_mcp_reply_declares_a_preference_it_could_not_apply(
        dated, tmp_path, monkeypatch):
    """The report's second half: the reply carried no sign of it.

    Built on a package with dates and no meaning index, which is the
    combination that misleads: everything else the caller can see says the
    preference was applied.
    """
    import asyncio
    import importlib
    import json

    folder = tmp_path / "docs"
    (folder / "extra.md").write_text(
        "---\ntitle: E\ndated: 2001-02-03\n---\n\nDrip irrigation laterals.\n",
        encoding="utf-8")
    target = tmp_path / "served.mdcx"
    archive.pack(folder, target, "k")

    monkeypatch.setenv("MDCX_FILE", str(target))
    monkeypatch.setenv("MDCX_KEY", "k")
    from mdcx import mcp_server
    importlib.reload(mcp_server)
    server = mcp_server.create_server()

    def reply(name: str, arguments: dict) -> dict:
        output = asyncio.run(server.call_tool(name, arguments))
        assert not output.is_error, output.content
        return json.loads(output.content[0].text)

    asked = reply("search", {"query": "drip irrigation", "prefer": "recent"})
    assert asked["prefer_applied"] is False
    assert asked["prefer_reason"]

    # And silent when nobody asked, so the field means something when it is there.
    plain = reply("search", {"query": "drip irrigation"})
    assert "prefer_applied" not in plain

    # The count the command line has always printed, so a caller can tell in
    # advance whether the preference has anything to work with.
    assert reply("info", {})["dated"]["documents"] > 0
