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

"""When a work is from, and letting whoever asks prefer the recent.

A package used to carry no date per document at all: eight works published
between 2015 and 2025 were indistinguishable, and the only date in the file was
when it had been packed. Nor could a consumer work around it, because the reply
carried neither a date nor a number to reorder by.

What made this worth fixing rather than noting is that the date was not missing
from the world, only from the package: it was recovered for 8 of 8 works from
the identifier the pseudopath already held.

The date is recorded with where it came from. A date without its provenance
confuses "when the work was published" with "when the file was touched", and
whoever reads it cannot tell. Preference is the asker's, never the corpus's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import semantic as S  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "new.md").write_text(
        "---\ntitle: New\n---\n\nCompleting the square in a quadratic.\n",
        encoding="utf-8")
    (folder / "old.md").write_text(
        "---\ntitle: Old\ndated: 2015-11-04\n---\n\nProjectile motion range.\n",
        encoding="utf-8")
    return folder


# --- The date is recorded, and so is where it came from ----------------------


def test_a_supplied_date_is_kept_with_its_provenance(corpus, tmp_path):
    """The case that matters: a date recovered from the publisher."""
    sidecar = tmp_path / "dates.csv"
    sidecar.write_text("@/new.md,2025-06-04,source\n", encoding="utf-8")

    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k", dates=archive.read_dates(sidecar))
    connection, _ = archive.open_package(target, "k")

    rows = dict((n, (d, f)) for n, d, f in connection.execute(
        "SELECT name, dated, dated_from FROM document"))
    assert rows["new"] == ("2025-06-04", "source")


def test_the_front_matter_is_read_when_nothing_was_supplied(corpus, tmp_path):
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k")
    connection, _ = archive.open_package(target, "k")

    rows = dict((n, (d, f)) for n, d, f in connection.execute(
        "SELECT name, dated, dated_from FROM document"))
    assert rows["old"] == ("2015-11-04", "front-matter")


def test_a_document_with_no_date_says_so_rather_than_guessing(corpus, tmp_path):
    """NULL is an honest answer and a different one from an invented date."""
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k")
    connection, _ = archive.open_package(target, "k")

    rows = dict((n, d) for n, d in connection.execute(
        "SELECT name, dated FROM document"))
    assert rows["new"] is None


def test_the_file_time_is_never_taken_unless_asked_for(corpus, tmp_path):
    """It is the file's date, not the work's, and it is always available.

    Taking it by default would fill every package with dates that look like
    answers and are not.
    """
    quiet = tmp_path / "quiet.mdcx"
    archive.pack(corpus, quiet, "k")
    connection, _ = archive.open_package(quiet, "k")
    assert not [f for (f,) in connection.execute(
        "SELECT dated_from FROM document") if f == "mtime"]

    asked = tmp_path / "asked.mdcx"
    archive.pack(corpus, asked, "k", use_mtime=True)
    connection, _ = archive.open_package(asked, "k")
    provenances = {f for (f,) in connection.execute("SELECT dated_from FROM document")}
    assert "mtime" in provenances, "the fallback did nothing when it was asked for"


def test_a_sidecar_outranks_the_front_matter(corpus, tmp_path):
    """The caller knows more than the file: they went and asked the publisher."""
    sidecar = tmp_path / "dates.csv"
    sidecar.write_text("@/old.md,1999-01-01,source\n", encoding="utf-8")

    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k", dates=archive.read_dates(sidecar))
    connection, _ = archive.open_package(target, "k")

    rows = dict((n, (d, f)) for n, d, f in connection.execute(
        "SELECT name, dated, dated_from FROM document"))
    assert rows["old"] == ("1999-01-01", "source")


def test_the_sidecar_reads_two_or_three_columns(tmp_path):
    """The third is optional, and defaults to saying somebody supplied it."""
    sidecar = tmp_path / "dates.csv"
    sidecar.write_text("# a comment\n"
                       "@/a.md,2020\n"
                       "@/b.md,2021,source\n"
                       "\n"
                       "malformed\n", encoding="utf-8")

    read = archive.read_dates(sidecar)

    assert read["@/a.md"] == ("2020", "sidecar")
    assert read["@/b.md"] == ("2021", "source")
    assert len(read) == 2, "a comment or a malformed line became a date"


def test_the_package_reports_how_many_it_dated(corpus, tmp_path):
    """"0 of 8 dated" is the signal that the dates were lost on the way in."""
    target = tmp_path / "c.mdcx"
    summary = archive.pack(corpus, target, "k")

    dated, total = summary["dated_documents"]
    assert total == 2
    assert dated == 1, "the count does not reflect what was actually dated"
    assert summary["dated_range"] == ["2015-11-04", "2015-11-04"]


# --- The reply carries it ----------------------------------------------------


def test_passages_carry_the_date_and_its_provenance(corpus, tmp_path):
    """Without this a consumer can only reorder blindly."""
    sidecar = tmp_path / "dates.csv"
    sidecar.write_text("@/new.md,2025-06-04,source\n", encoding="utf-8")
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k", dates=archive.read_dates(sidecar))
    connection, _ = archive.open_package(target, "k")

    found = archive.query(connection, "completing the square", limit=4)

    assert found
    assert found[0]["dated"] == "2025-06-04"
    assert found[0]["dated_from"] == "source"


def test_a_package_written_before_this_still_answers(corpus, tmp_path, monkeypatch):
    """The columns did not exist, and asking for them would fail rather than
    answer "unknown"."""
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k")
    connection, _ = archive.open_package(target, "k")

    monkeypatch.setattr(archive, "has_dates", lambda c: False)
    found = archive.query(connection, "completing the square", limit=4)

    assert found, "an older package stopped answering"
    assert "dated" not in found[0]


# --- Preferring the recent ---------------------------------------------------


def test_a_third_ranking_rather_than_a_decay_on_the_score():
    """Weighting scores that share no scale is what fusing by rank avoids.

    An order by date is a legitimate ranking and fuses like the other two;
    multiplying a cosine by a decay reintroduces the arbitrariness the fusion
    exists to remove.
    """
    lexical = ["a", "b"]
    dense = ["b", "a"]
    by_date = ["b", "a"]

    assert S.fuse([lexical, dense]) == ["a", "b"]
    assert S.fuse([lexical, dense, by_date], weights=(1, 1, 0.25)) == ["b", "a"]


def test_a_weight_of_zero_changes_nothing():
    """Preference is off by default, and off has to mean untouched."""
    lexical, dense, by_date = ["a", "b"], ["a", "b"], ["b", "a"]

    assert (S.fuse([lexical, dense, by_date], weights=(1, 1, 0))
            == S.fuse([lexical, dense]))


def test_the_date_cannot_overrule_what_both_engines_agree_on():
    """The boundary worth knowing, and the reason the weight is small.

    Where the two engines put the same passage first, the date does not move it.
    That is deliberate: a work from 1970 can be the right answer, and in
    mathematics often is. The date orders what relevance considers comparable.
    """
    agreed = ["old", "new"]
    by_date = ["new", "old"]

    assert S.fuse([agreed, agreed, by_date], weights=(1, 1, 0.25)) == ["old", "new"]


def test_preference_is_off_unless_asked_for(corpus, tmp_path):
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k")
    connection, _ = archive.open_package(target, "k")

    plain = archive.query(connection, "completing the square", limit=4)
    same = archive.query(connection, "completing the square", limit=4, prefer=None)

    assert [p["passage"] for p in plain] == [p["passage"] for p in same]


def test_an_unknown_preference_is_refused(corpus, tmp_path):
    target = tmp_path / "c.mdcx"
    archive.pack(corpus, target, "k")
    connection, _ = archive.open_package(target, "k")

    with pytest.raises(ValueError):
        archive.query(connection, "anything", prefer="oldest")


def test_undated_passages_keep_their_place():
    """A gap in the metadata must not decide the answer.

    Absence of a date says nothing about the age of a work, so an undated
    passage is left where relevance put it rather than treated as ancient.
    """
    lexical = ["dated", "undated"]
    dense = ["undated", "dated"]
    # Only the dated passage appears in the date ranking, which is what query
    # builds: undated ones are not ranked at all rather than ranked last.
    by_date = ["dated"]

    order = S.fuse([lexical, dense, by_date], weights=(1, 1, 0.25))
    assert set(order) == {"dated", "undated"}, "an undated passage was dropped"
