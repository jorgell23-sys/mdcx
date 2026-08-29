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
