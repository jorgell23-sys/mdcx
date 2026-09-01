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

"""What whoever writes a source should not have to work out again.

mdcx does not fetch, and that is not changing: talking to a particular catalogue
belongs with whoever knows it, and duplicating that here would make it neither
better nor more correct.

What does not belong there is the part that has nothing to do with any
catalogue. The first real plugin written against this contract fell into three
traps the module's own docstring already names -- so the knowledge was written
down and reading it was not enough. The expensive one returned bytes without
failing: 5,384 of them, for a book, and they were the cover thumbnail, because
DSpace names it `9789819647453.pdf.jpg`. The rule being applied, "read the file
name rather than the declared type", is right and does not say where the name
ends.

`ReferenceSource` below is the fourth thing the report asked for: an executable
example, which does not go quietly out of date the way a paragraph does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import sources  # noqa: E402

PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n"
# What the catalogue actually returned for a book, in shape: a JPEG.
THUMBNAIL = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 200


class ReferenceSource:
    """A source in twenty lines, against nothing.

    It exists to be read. Everything a real one adds is catalogue knowledge:
    which endpoint lists works, which field holds the download, that one column
    answers 403 and needs the handle resolved separately.

    What it shows is the shape. `search` is cheap and narrows; `fetch` is
    expensive and commits; an identifier is opaque to mdcx and is whatever the
    source needs to find the work again; and `fetch` raises rather than
    returning something that is not the document.
    """

    name = "reference"

    def __init__(self, holdings: dict[str, bytes] | None = None):
        self.holdings = holdings or {
            "ref:1": PDF, "ref:2": PDF + b"second"}

    def search(self, query: str, limit: int) -> list[sources.Candidate]:
        found = [
            sources.Candidate(identifier=key, title=f"A work about {query}",
                              source=self.name, summary="Held for the example.",
                              # Said here because the catalogue knows it here,
                              # and a caller deciding what to fetch wants to
                              # know which server it would be asking. Two
                              # hosts rather than one, because spreading the
                              # requests is the point.
                              download=f"https://{key.split(':')[1]}.example/"
                                       f"{key.replace(':', '-')}.pdf")
            for key in sorted(self.holdings)
        ]
        return found[:limit]

    def fetch(self, candidate: sources.Candidate, timeout: int) -> bytes:
        try:
            return self.holdings[candidate.identifier]
        except KeyError as missing:
            # Raising, rather than returning empty bytes or a landing page.
            raise LookupError(
                f"{candidate.identifier} is not in this catalogue") from missing


# --- Reading four bytes ------------------------------------------------------


def test_the_thumbnail_is_not_a_pdf():
    """The measured mistake, and the one check that makes it impossible."""
    assert sources.looks_like(PDF, "pdf")
    assert not sources.looks_like(THUMBNAIL, "pdf")


def test_the_answer_says_what_it_was():
    """"This is not a PDF" is not actionable; "this is a JPEG" is."""
    assert sources.identify(THUMBNAIL) == "jpeg"
    assert sources.identify(PDF) == "pdf"
    assert sources.identify(b"nothing in particular") is None


def test_a_container_format_is_recognised_as_its_container():
    """docx, xlsx, pptx and epub are zips, and four bytes cannot say which.

    Reported as what it is rather than refused: what this exists to rule out is
    bytes that are not the document at all.
    """
    zipped = b"PK\x03\x04" + b"\x00" * 40

    assert sources.looks_like(zipped, "docx")
    assert sources.looks_like(zipped, "epub")
    assert sources.identify(zipped) == "zip"


def test_asking_about_a_format_mdcx_does_not_convert_is_answered(algebra=None):
    """False rather than an error, so a caller need not guard the call."""
    assert not sources.looks_like(PDF, "wav")
    assert not sources.looks_like(b"", "pdf")


# --- Waiting when a server asks ----------------------------------------------


def test_it_waits_as_long_as_the_server_asked():
    """`Retry-After` is standard, and honouring it is not catalogue knowledge."""
    waits: list[float] = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise sources.RateLimited(retry_after=7.0)
        return "the document"

    got = sources.patiently(call, sleep=waits.append)

    assert got == "the document"
    assert waits == [7.0, 7.0], "the delay the server asked for was not used"


def test_it_backs_off_on_its_own_when_the_server_said_nothing():
    waits: list[float] = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] < 4:
            raise sources.RateLimited()
        return "ok"

    sources.patiently(call, wait=1.0, sleep=waits.append)

    assert waits == [1.0, 2.0, 4.0]


def test_it_does_not_retry_what_will_not_change():
    """A 403 on a whole download column is not transient, and retrying it
    spends the caller's time to learn what the first attempt already said."""
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        raise PermissionError("403")

    with pytest.raises(PermissionError):
        sources.patiently(call, sleep=lambda _: None)

    assert attempts["n"] == 1


def test_running_out_of_patience_raises_rather_than_returning_nothing():
    def call():
        raise sources.RateLimited(retry_after=1.0)

    with pytest.raises(sources.RateLimited):
        sources.patiently(call, attempts=3, sleep=lambda _: None)


# --- Checking a plugin against the contract ----------------------------------


def test_the_reference_source_conforms():
    assert sources.conforms(ReferenceSource()) == []


def test_a_source_returning_a_thumbnail_is_caught():
    """The error the report says a checker would have shown on the first run."""
    class Cover(ReferenceSource):
        def fetch(self, candidate, timeout):
            return THUMBNAIL

    problems = sources.conforms(Cover())

    assert problems
    assert any("jpeg" in p and "thumbnail" in p for p in problems), problems


def test_a_source_returning_a_landing_page_is_caught():
    """An error the server sent with a 200 is the other silent one."""
    class Landing(ReferenceSource):
        def fetch(self, candidate, timeout):
            return b"<!DOCTYPE html><html><body>Not found</body></html>"

    assert any("HTML" in p for p in sources.conforms(Landing()))


def test_an_empty_identifier_is_caught_before_anything_is_fetched():
    """A candidate that cannot be fetched again is useless downstream."""
    class Nameless(ReferenceSource):
        def search(self, query, limit):
            return [sources.Candidate(identifier="", title="A work")]

    assert any("empty identifier" in p for p in sources.conforms(Nameless()))


def test_returning_more_than_the_limit_is_caught():
    class Greedy(ReferenceSource):
        def search(self, query, limit):
            return super().search(query, limit + 5)

    assert any("for a limit of" in p for p in sources.conforms(Greedy(), limit=1))


def test_raising_from_fetch_is_allowed_and_only_noted():
    """A work that cannot be had is a fact about the catalogue, not a fault."""
    class Absent(ReferenceSource):
        def fetch(self, candidate, timeout):
            raise LookupError("withdrawn")

    problems = sources.conforms(Absent())

    assert problems
    assert all(p.startswith("note:") for p in problems), problems


def test_a_catalogue_holding_nothing_is_not_a_failing_plugin():
    """Checking the holdings rather than the plugin would be the wrong test."""
    class Empty(ReferenceSource):
        def search(self, query, limit):
            return []

    problems = sources.conforms(Empty())

    assert all(p.startswith("note:") for p in problems), problems


def test_something_that_is_not_a_source_says_so_and_stops():
    problems = sources.conforms(object())

    assert any("search() and fetch()" in p for p in problems)


# --- The command line ---------------------------------------------------------


def test_the_check_reports_conformance(capsys, monkeypatch):
    monkeypatch.setattr(sources, "require",
                        lambda names=None: {"reference": ReferenceSource()})

    code = sources._check(["reference"])

    assert code == 0
    assert "conforms" in capsys.readouterr().out


def test_the_check_fails_on_a_fault_and_not_on_a_note(capsys, monkeypatch):
    class Cover(ReferenceSource):
        def fetch(self, candidate, timeout):
            return THUMBNAIL

    monkeypatch.setattr(sources, "require", lambda names=None: {"c": Cover()})
    assert sources._check(["c"]) == 1

    class Absent(ReferenceSource):
        def fetch(self, candidate, timeout):
            raise LookupError("withdrawn")

    monkeypatch.setattr(sources, "require", lambda names=None: {"a": Absent()})
    assert sources._check(["a"]) == 0, "a note must not be reported as a fault"


def test_nothing_installed_is_told_apart_from_a_plugin_that_fails(capsys,
                                                                  monkeypatch):
    def refuse(names=None):
        raise sources.NoSourcesInstalled("no catalogue is installed")

    monkeypatch.setattr(sources, "require", refuse)

    assert sources._check([]) == 2, (
        "an incomplete installation is not a failing plugin")


# --- And the line that is not crossed ----------------------------------------


def test_mdcx_still_ships_no_catalogue_and_needs_no_network():
    """The kit is local by construction: bytes in memory, a wait, a contract.

    If any of it ever reaches for the network, this is where that shows.
    """
    text = (Path(__file__).resolve().parents[1]
            / "src" / "mdcx" / "sources.py").read_text(encoding="utf-8")

    for reaching in ("import requests", "import urllib", "import http",
                     "httpx", "socket"):
        assert reaching not in text, f"sources.py reaches for {reaching}"


# --- Run as a module, which is how the report ran it --------------------------


def test_the_check_agrees_with_itself_however_it_is_invoked(monkeypatch):
    """The same function answered differently depending on how it was called.

    `python -m mdcx.sources` executes sources.py under the name `__main__`, so
    every class it defines is created a second time. `Candidate` then exists
    twice -- same name, same code, different identity -- and a plugin builds the
    one it imported, because a plugin has no other way to build one. `isinstance`
    compared the two and answered no, so the checker reported "search()[0] is
    Candidate, not a Candidate": literally true and useless.

    Reproduced here with runpy, which enters the module exactly as `-m` does.
    Patching the imported module and having the run see it is the whole point:
    if the entry point stops delegating, it reads its own copy instead and this
    fails.
    """
    import runpy

    monkeypatch.setattr(sources, "require",
                        lambda names=None: {"reference": ReferenceSource()})
    monkeypatch.setattr(sys, "argv", ["mdcx.sources"])

    with pytest.raises(SystemExit) as left:
        runpy.run_module("mdcx.sources", run_name="__main__")

    assert left.value.code == 0, (
        "run as a module it disagreed with the same check called as a function")


def test_a_candidate_is_a_candidate_whichever_module_object_holds_it():
    """The invariant under the bug, stated so the shape stays visible.

    A checker that invents problems is worse than none: the first thing its
    reader does is doubt their own plugin.
    """
    made_by_a_plugin = sources.Candidate(identifier="1", title="A work")

    assert sources.conforms(ReferenceSource()) == []
    assert isinstance(made_by_a_plugin, sources.Candidate)


def test_the_command_line_offers_help(capsys):
    """It ran the checker instead, and ignored the argument in silence."""
    with pytest.raises(SystemExit) as left:
        sources._check(["--help"])

    assert left.value.code == 0
    printed = capsys.readouterr().out
    assert "--check" in printed and "--no-fetch" in printed


def test_check_names_a_source_either_way(monkeypatch):
    """`--check doab` is what the proposal documented; a bare name also works."""
    asked: list = []
    monkeypatch.setattr(sources, "require",
                        lambda names=None: (asked.append(names),
                                            {"reference": ReferenceSource()})[1])

    sources._check(["--check", "reference"])
    sources._check(["reference"])

    assert asked == [["reference"], ["reference"]]


# --- Which server the file would come from ------------------------------------


def test_a_candidate_can_say_where_the_file_would_come_from():
    """The split between a cheap search and an expensive fetch exists so a
    caller can decide what is worth fetching, and which server it would be
    asking is one of the things worth deciding on.

    Measured over 40 candidates from one catalogue: 25 of them, 62 per cent,
    resolve to a single host that answers 403 to everything, while another
    returns a 5 MB PDF in two seconds. A caller working down the ranking spends
    its budget on the first and never reaches the second.
    """
    knows = sources.Candidate(
        identifier="1", title="A work",
        download="https://library.oapen.org/bitstream/20.500/1/a.pdf")

    assert knows.host == "library.oapen.org"


def test_not_knowing_the_host_is_said_as_not_knowing():
    """Empty is different from there being none, which is why it is not derived
    from `url`: that one is usually the record's page, not the file."""
    assert sources.Candidate(identifier="1", title="A").host == ""
    assert sources.Candidate(identifier="1", title="A",
                             url="https://doab.org/handle/1").host == ""
    assert sources.Candidate(identifier="1", title="A",
                             download="not a url at all").host == ""


def test_the_check_says_when_one_host_holds_the_catalogue():
    """Not a fault -- it is how that catalogue is built -- and the single most
    useful thing to know before spending a download budget on it."""
    class OneHost(ReferenceSource):
        def search(self, query, limit):
            return [sources.Candidate(
                identifier=str(i), title=f"work {i}",
                download=f"https://library.oapen.org/bitstream/{i}.pdf")
                for i in range(3)][:limit]

    problems = sources.conforms(OneHost(), fetch=False)

    assert any("library.oapen.org" in p and p.startswith("note:")
               for p in problems), problems


def test_saying_nothing_about_hosts_is_itself_worth_a_note():
    """A note and not a fault: a catalogue may genuinely not know until it
    asks, and then the honest answer is an empty field."""
    class Silent(ReferenceSource):
        def search(self, query, limit):
            return [sources.Candidate(identifier="1", title="A work")]

    problems = sources.conforms(Silent(), fetch=False)

    assert any("which host" in p for p in problems), problems
    assert all(p.startswith("note:") for p in problems), problems
