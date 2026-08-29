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

"""Where works come from, and why no catalogue lives inside mdcx.

mdcx converts, packages and answers, and depends on no network of its own. The
contract here is what lets it stay that way while still being able to build a
corpus: a source is a plugin, found through an entry point.

Keeping the adapters out is deliberate. What looks like a simple HTTP client is
not -- one catalogue answers 403 to its whole download column and needs its
handle resolved separately, another returns the same scrape cursor for every
page -- and that knowledge belongs with whoever has it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import sources  # noqa: E402


class Toy:
    """A catalogue that meets the contract without touching a network."""

    name = "toy"

    def __init__(self, works=None):
        self.works = works or [
            sources.Candidate(identifier="1", title="Water and policy",
                              summary="On managing a scarce resource.",
                              source="toy", language="en")]

    def search(self, query, limit):
        return self.works[:limit]

    def fetch(self, candidate, timeout):
        return b"%PDF-1.4"


def test_the_contract_is_checkable():
    """A plugin can be told it conforms before anything depends on it."""
    assert isinstance(Toy(), sources.Source)


def test_something_missing_an_operation_does_not_conform():
    class HalfDone:
        name = "half"

        def search(self, query, limit):
            return []

    assert not isinstance(HalfDone(), sources.Source)


def test_mdcx_ships_no_catalogue_of_its_own():
    """The property worth keeping: nothing here reaches a network.

    If a catalogue were ever added to the package itself, this is where that
    decision would show up rather than being discovered later.
    """
    assert sources.available() == {}, (
        "a source is installed with mdcx itself, which makes the package "
        "depend on a network it was built not to need")


def test_asking_with_nothing_installed_says_so_precisely(monkeypatch):
    """Not an empty result: an empty result is a question nobody can act on."""
    monkeypatch.setattr(sources, "available", dict)

    with pytest.raises(sources.NoSourcesInstalled) as refused:
        sources.require()

    said = str(refused.value)
    assert sources.ENTRY_POINT_GROUP in said, (
        "the message does not say how a catalogue is installed")
    assert "already exists" in said, (
        "it does not say that answering over an existing package needs none")


def test_naming_one_that_is_not_there_is_a_different_answer(monkeypatch):
    """Told which of the two happened: none installed, or none by that name."""
    monkeypatch.setattr(sources, "available", lambda: {"toy": Toy()})

    with pytest.raises(sources.NoSourcesInstalled) as refused:
        sources.require(["absent"])

    said = str(refused.value)
    assert "absent" in said and "toy" in said, (
        "the message names neither what was asked for nor what there is")


def test_named_sources_come_back_in_the_order_asked(monkeypatch):
    monkeypatch.setattr(sources, "available",
                        lambda: {"a": Toy(), "b": Toy(), "c": Toy()})

    assert list(sources.require(["c", "a"])) == ["c", "a"]


def test_a_broken_plugin_does_not_take_the_others_with_it(monkeypatch):
    """A corpus built from the rest is worth more than no corpus at all."""
    class Entry:
        def __init__(self, name, load):
            self.name = name
            self.load = load

    def entry_points(group):
        return [Entry("broken", lambda: (_ for _ in ()).throw(ImportError("nope"))),
                Entry("toy", lambda: Toy)]

    import importlib.metadata as meta
    monkeypatch.setattr(meta, "entry_points", entry_points)

    found = sources.available()

    assert list(found) == ["toy"], "one broken plugin hid the working one"


def test_a_candidate_packs_as_a_record():
    """The bridge to --from-jsonl: a catalogue is searched by being packed."""
    candidate = Toy().search("water", 1)[0]
    record = candidate.as_record()

    assert record["name"] and record["text"]
    assert candidate.title in record["text"]
    assert candidate.summary in record["text"], (
        "the summary is what a catalogue search reads, and it was dropped")


def test_a_candidate_with_no_summary_still_packs():
    """Not every catalogue has one, and a title alone is still searchable."""
    bare = sources.Candidate(identifier="7", title="A title alone")
    assert bare.as_record()["text"] == "A title alone"


def test_a_candidate_with_no_title_falls_back_to_its_identifier():
    """A record with no name cannot be cited, which is worse than an ugly one."""
    bare = sources.Candidate(identifier="oapen:12345", title="")
    assert bare.as_record()["name"] == "oapen:12345"
