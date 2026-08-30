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

"""Where works come from, for the part of the work mdcx does not do.

mdcx converts, packages and answers. It does not fetch, and this module is what
lets it not fetch while still being able to: the contract a catalogue has to
meet, and the means of finding whichever ones are installed. No adapter lives
here, and mdcx depends on no network of its own.

That is deliberate rather than minimal. What looks like a simple HTTP client is
not: one catalogue answers 403 to its whole download column and needs its handle
resolved through a separate call, another returns the same scrape cursor for
every page so paging has to go by key, and a handle prefix has three parts, so
the obvious pattern for it cuts in the wrong place. That work belongs with
whoever knows the catalogue, and duplicating it here would make it neither
better nor more correct.

So a source is a plugin. It is found through the ``mdcx.sources`` entry point
group, which means installing a package is all it takes:

    [project.entry-points."mdcx.sources"]
    oapen = "my_package.oapen:Catalogue"

Answering a question over a package that already exists needs none of this. Only
building a new corpus does, and where nothing is installed it says exactly that
rather than failing obscurely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "mdcx.sources"


@dataclass
class Candidate:
    """One work a source offers, before anything has been downloaded.

    Enough to decide whether it is worth fetching, and no more: the deciding is
    done over these, and fetching is what costs. ``identifier`` is whatever the
    source needs to fetch it again and is opaque here.
    """

    identifier: str
    title: str
    source: str = ""
    summary: str = ""
    language: str = ""
    url: str = ""
    extra: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        """The candidate as a record for packing, so a catalogue is searchable.

        Title and summary are what a catalogue search reads, and putting them
        in one field is what lets the same passage carry both.
        """
        text = self.title if not self.summary else f"{self.title}\n\n{self.summary}"
        return {"name": self.title[:120] or self.identifier,
                "text": text,
                "folder": self.source or "catalogue",
                "source": "OTHER"}


@runtime_checkable
class Source(Protocol):
    """A catalogue mdcx can ask, implemented outside mdcx.

    Two operations, because they cost differently and are decided separately:
    searching is cheap and narrows, fetching is expensive and commits.
    """

    name: str

    def search(self, query: str, limit: int) -> list[Candidate]:
        """Works this catalogue offers for a query, most promising first."""
        ...

    def fetch(self, candidate: Candidate, timeout: int) -> bytes:
        """The document itself, as bytes. Raises if it cannot be had."""
        ...


class NoSourcesInstalled(RuntimeError):
    """Raised where building a corpus is asked for and nothing can fetch.

    Its own type, because it is not a failure of the request: the request is
    fine and the installation is incomplete, and those want different answers
    from whoever is calling.
    """


def available() -> dict[str, Source]:
    """Every source this installation can use, by name.

    A source that fails to load is skipped rather than fatal: one broken plugin
    should not stop the others, and a corpus built from the rest is worth more
    than no corpus at all.
    """
    from importlib import metadata

    found: dict[str, Source] = {}
    try:
        entries = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - an installation that cannot be read has none
        return found

    for entry in entries:
        try:
            loaded = entry.load()
            source = loaded() if isinstance(loaded, type) else loaded
        except Exception:  # noqa: BLE001
            continue
        if isinstance(source, Source):
            found[getattr(source, "name", entry.name)] = source
    return found


def require(names: list[str] | None = None) -> dict[str, Source]:
    """The sources asked for, or every one installed, refusing to guess.

    Silence would be the wrong answer here. Building a corpus with no source is
    not an empty corpus, it is a question nobody can act on, so it says which
    of the two happened: none installed, or none matching what was named.
    """
    found = available()
    if not found:
        raise NoSourcesInstalled(
            "no catalogue is installed. mdcx converts, packages and answers; "
            f"fetching is done by a plugin declaring a '{ENTRY_POINT_GROUP}' "
            "entry point. Answering questions over a package that already "
            "exists needs none of them.")
    if not names:
        return found

    missing = [n for n in names if n not in found]
    if missing:
        raise NoSourcesInstalled(
            f"no catalogue named {', '.join(missing)}. Installed: "
            f"{', '.join(sorted(found)) or 'none'}")
    return {n: found[n] for n in names}


# --- What a plugin should not have to write again -----------------------------
#
# None of this fetches, and none of it knows a catalogue. That is the line: the
# work of talking to a particular catalogue belongs with whoever knows it, and
# checking bytes already in memory, waiting when a server asks, and testing that
# a plugin keeps its own contract are none of them about a catalogue.


# What a file begins with, for the formats mdcx converts. Only signatures that
# identify a format outright: a check answering "maybe" would be worse than no
# check, because the caller would then have to decide what to do about maybe.
#
# ZIP appears once and stands for several things -- docx, xlsx, pptx and epub
# are zip containers -- so asking whether bytes look like a docx says they are a
# zip and not which zip. That is the honest limit of four bytes, and it is the
# case this exists for anyway: a catalogue handing back something that is not
# the document at all.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF",),
    "zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "gif": (b"GIF87a", b"GIF89a"),
    "tiff": (b"II*\x00", b"MM\x00*"),
    "html": (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"),
    "gzip": (b"\x1f\x8b",),
    "rtf": (b"{\\rtf",),
}
_ZIP_FORMATS = {"docx", "xlsx", "pptx", "epub", "odt", "ods", "odp"}


def looks_like(data: bytes, kind: str) -> bool:
    """Whether these bytes begin the way that kind of file begins.

    The check that makes the expensive mistake impossible, and it needs neither
    the network nor any knowledge of a catalogue: it reads bytes already in
    memory.

    The mistake is not hypothetical and did not fail loudly. A catalogue
    returned 5,384 bytes for a book without raising, and they were the cover
    thumbnail: the attachment was named `9789819647453.pdf.jpg`, which is how
    DSpace names one. The rule being applied -- "the declared type is useless,
    read the file name" -- is right and is not enough. What was missing was not
    whether to read the name but where the name ends.

    An unknown kind is False rather than an error, so a caller may ask about a
    format mdcx does not convert and still get a usable answer.
    """
    if not data:
        return False
    wanted = kind.lower().lstrip(".")
    if wanted in _ZIP_FORMATS:
        wanted = "zip"
    elif wanted == "jpg":
        wanted = "jpeg"
    return any(bytes(data).startswith(s) for s in _SIGNATURES.get(wanted, ()))


def identify(data: bytes) -> str | None:
    """What these bytes look like, or None when nothing matches.

    For the message rather than the decision: knowing the answer was a JPEG is
    what turns "this is not a PDF" into something the reader can act on.
    """
    if not data:
        return None
    for kind, signatures in _SIGNATURES.items():
        if any(bytes(data).startswith(s) for s in signatures):
            return kind
    return None


class RateLimited(RuntimeError):
    """Raised by a source when a catalogue asks it to wait.

    Its own type so that waiting can be told from failing, and it carries the
    delay the server asked for when the server said one: `Retry-After` is a
    standard header, and honouring it is not catalogue knowledge.
    """

    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message or "the catalogue asked us to wait")
        self.retry_after = retry_after


def patiently(call, attempts: int = 4, wait: float = 2.0,
              ceiling: float = 60.0, sleep=None):
    """Run something that may be rate limited, waiting when asked to.

    Every catalogue rate limits -- measured on one, sixteen of twenty
    consecutive fetches came back 429 -- so every plugin would otherwise write
    this loop. Nothing in it is about any catalogue: it retries what raises
    `RateLimited`, waits as long as the server asked when the server said, and
    doubles its own wait when it did not.

    It catches nothing else, deliberately. A 403 on a whole download column is
    not a transient condition, and retrying it spends the caller's time to learn
    what the first attempt already said. That is the difference between waiting
    and hoping.

    The last attempt raises rather than returning something wrong, so a caller
    that ran out of patience knows that is what happened.
    """
    import time

    rest = sleep or time.sleep
    delay = wait
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except RateLimited as limited:
            if attempt >= attempts:
                raise
            asked = limited.retry_after
            rest(min(ceiling, asked if asked and asked > 0 else delay))
            delay = min(ceiling, delay * 2)


def conforms(source, query: str = "graph theory", limit: int = 3,
             timeout: int = 60, fetch: bool = True) -> list[str]:
    """Check a source against the contract and return what it got wrong.

    An empty list means it conforms. What is checked is only what the contract
    promises, because that is all mdcx can know: a catalogue is entitled to hold
    nothing for a query, and calling that a failure would be testing its
    holdings rather than the plugin.

    Lines beginning `note:` are observations rather than faults -- things that
    may be correct and are worth a second look.

    A plugin had nothing to check itself against, and what that cost was
    measured: the first thing this reports is a source returning a cover
    thumbnail as though it were the book.
    """
    problems: list[str] = []
    name = getattr(source, "name", None)
    if not isinstance(name, str) or not name:
        problems.append("name: missing or empty, and it is how a source is addressed")

    if not isinstance(source, Source):
        problems.append("the object does not offer both search() and fetch()")
        return problems

    try:
        found = source.search(query, limit)
    except Exception as e:  # noqa: BLE001
        problems.append(f"search() raised {type(e).__name__}: {e}")
        return problems

    if not isinstance(found, list):
        problems.append(f"search() returned {type(found).__name__}, not a list")
        return problems
    if len(found) > limit:
        problems.append(f"search() returned {len(found)} for a limit of {limit}")

    for i, candidate in enumerate(found):
        if not isinstance(candidate, Candidate):
            problems.append(
                f"search()[{i}] is {type(candidate).__name__}, not a Candidate")
            continue
        if not candidate.identifier:
            problems.append(
                f"search()[{i}] has an empty identifier, so it cannot be fetched")
        if not candidate.title:
            problems.append(f"search()[{i}] has an empty title")

    # Fetching costs, so it is one candidate, and it can be turned off.
    usable = [c for c in found if isinstance(c, Candidate) and c.identifier]
    if fetch and usable:
        try:
            data = source.fetch(usable[0], timeout)
        except Exception as e:  # noqa: BLE001
            # Raising is permitted -- the contract says so -- and is reported as
            # an observation, because a work that cannot be had is a fact about
            # the catalogue rather than a fault in the plugin.
            problems.append(
                f"note: fetch() raised {type(e).__name__}: {e}. That is "
                "allowed; check it is not raising for everything")
        else:
            if not isinstance(data, (bytes, bytearray)):
                problems.append(
                    f"fetch() returned {type(data).__name__}, not bytes")
            elif not data:
                problems.append("fetch() returned no bytes and did not raise")
            else:
                kind = identify(bytes(data))
                if kind in ("jpeg", "png", "gif", "tiff"):
                    problems.append(
                        f"fetch() returned {len(data)} bytes of {kind}. A "
                        "catalogue naming a cover thumbnail after the book -- "
                        "1234.pdf.jpg -- yields exactly this, silently")
                elif kind == "html":
                    problems.append(
                        f"fetch() returned {len(data)} bytes of HTML, which is "
                        "usually a landing page, or an error the server sent "
                        "with a 200")
                elif kind is None:
                    problems.append(
                        f"note: fetch() returned {len(data)} bytes of no "
                        "recognised type. It may be right; check it")

    if not found:
        problems.append(
            "note: search() returned nothing, so fetch() was not exercised. "
            "Try a query this catalogue holds")
    return problems


def _check(argv: list[str]) -> int:
    """`python -m mdcx.sources --check <name>`, for whoever writes a plugin."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m mdcx.sources",
        description="Check an installed source against the contract.")
    parser.add_argument("--check", action="append", metavar="NAME", dest="named",
                        help="the source to check. Repeat it, or omit it to "
                             "check every installed source")
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help="the same, given positionally")
    parser.add_argument("--query", default="graph theory",
                        help="what to search for. Use one this catalogue holds, "
                             "or search() returns nothing and fetch() is never "
                             "exercised")
    parser.add_argument("--no-fetch", action="store_true",
                        help="check search() alone. Fetching costs, and one "
                             "candidate is fetched otherwise")
    args = parser.parse_args(argv)

    query = args.query
    offline = args.no_fetch
    names = list(args.names) + list(args.named or [])

    try:
        sources = require(names or None)
    except NoSourcesInstalled as e:
        print(e, file=sys.stderr)
        return 2

    worst = 0
    for name, source in sorted(sources.items()):
        problems = conforms(source, query=query, fetch=not offline)
        faults = [p for p in problems if not p.startswith("note:")]
        notes = [p for p in problems if p.startswith("note:")]
        print(f"{name}: "
              + ("conforms" if not faults else f"{len(faults)} problem(s)"))
        for problem in faults:
            print(f"  - {problem}")
        for note in notes:
            print(f"  . {note[6:]}")
        if faults:
            worst = 1
    return worst


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    # Delegated to the imported module rather than run from here, and that is
    # the whole of it: `python -m mdcx.sources` executes this file under the
    # name `__main__`, so every class it defines is created a second time.
    # `Candidate` then exists twice -- same name, same code, different identity
    # -- and a plugin builds the one it imported from `mdcx.sources`, because a
    # plugin has no other way to build one.
    #
    # `isinstance` compared the two and answered no, so the checker reported
    # "search()[0] is Candidate, not a Candidate": literally true, and useless.
    # A checker that invents problems is worse than none, because the first
    # thing its reader does is doubt their own plugin.
    #
    # Importing here binds the canonical module -- this file under its real
    # name, already in sys.modules -- so the check runs against the classes
    # every plugin actually holds. Comparing by structure instead would have
    # hidden the symptom and left the duplication for the next comparison to
    # find.
    import sys

    from mdcx.sources import _check as _canonical

    raise SystemExit(_canonical(sys.argv[1:]))
