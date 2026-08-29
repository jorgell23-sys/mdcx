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

"""Model Context Protocol server exposing an ``.mdcx`` corpus to agents.

An agent answering questions about a document collection can either receive the
documents in its context window, which is expensive and bounded, or query a
component that already knows where each item is. This server provides the second
option: it receives a question, searches locally, and returns only the passages
that answer it, each with its provenance.

Configuration is supplied through environment variables so that the key is not
passed on the command line:

    MDCX_FILE=/path/to/corpus.mdcx
    MDCX_KEY=package-key

    python -m mdcx.mcp_server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import archive

_STATE: dict = {}


def _split_setting(value: str) -> list[str]:
    """Split a setting that may name several packages.

    The separator is the one the platform uses for lists of paths, so a Windows
    path keeps its drive letter. A comma is also accepted, since it is what
    people write.
    """
    parts: list[str] = []
    for piece in value.split(","):
        parts.extend(t for t in piece.split(os.pathsep) if t.strip())
    return [t.strip() for t in parts if t.strip()]


def _open_packages() -> list[dict]:
    """Open every configured package once and reuse them.

    Decryption and decompression take a fraction of a second and need not be
    repeated per query. Each database is held in memory, so an open package
    leaves no plaintext copy on disk.
    """
    if "packages" in _STATE:
        return _STATE["packages"]

    adjustment = os.environ.get("MDCX_FILE", "").strip()
    adjustment_keys = os.environ.get("MDCX_KEY", "")
    if not adjustment:
        raise RuntimeError("MDCX_FILE is not set: provide the path to the .mdcx package.")
    if not adjustment_keys:
        raise RuntimeError("MDCX_KEY is not set: the package is encrypted.")

    paths = _split_setting(adjustment)
    keys = _split_setting(adjustment_keys)
    # One key serves every package, which is the ordinary case; several keys are
    # matched to the packages in order.
    if len(keys) == 1:
        keys = keys * len(paths)
    if len(keys) != len(paths):
        raise RuntimeError(
            f"MDCX_KEY names {len(keys)} keys for {len(paths)} packages: "
            "give one key for all of them, or one key per package in the same order.")

    packages: list[dict] = []
    for path, key in zip(paths, keys):
        target = Path(path)
        if not target.is_file():
            raise RuntimeError(f"Package not found: {target}")
        connection, header = archive.open_package(target, key)
        packages.append({"name": target.name, "path": target,
                         "connection": connection, "header": header})

    _STATE["packages"] = packages
    _STATE["connection"] = packages[0]["connection"]
    _STATE["header"] = packages[0]["header"]
    return packages


def _connection():
    """The first package, for the operations that read a single record."""
    return _open_packages()[0]["connection"]



# How far below the best a package may sit and still be asked. Measured on
# packages that hold unrelated subjects: when one alone answers the query, the
# next best sits at least 0.18 below it, and when the subject genuinely spans
# two, the second sits within 0.04. The gate belongs between those two regimes.
RELEVANCE_MARGIN = 0.10


# Two conditions, and both are required, because each one alone has been shipped
# and each one alone was wrong.
#
# How near the best passage comes depends on what is in the collection: on one
# corpus the questions it answers start at 0.57 and on another at 0.64, so a
# threshold set on the first marks nothing on the second. That was shipped at
# 0.45 and never fired.
#
# How far the best passage stands clear of the fiftieth depends on how many
# passages there are: with a hundred thousand the fiftieth is nearly the best,
# with forty it is nearly the worst. That was shipped at 0.25 and marked
# everything.
#
# Requiring both is what neither gives on its own. A question is called
# unanswered only when the corpus comes no nearer than NOTHING_NEAR *and* its
# best passage does not rise out of the tail. On a small corpus the first
# condition fires wrongly and the second holds it back; on a large one the
# second fires wrongly and the first holds it back. Both values sit where the
# two groups separated on the corpus each was measured on.
#
# It is still two constants describing a collection they cannot see, which is
# the wrong shape for this. The right one is to measure the corpus when it is
# packed and carry the number in the manifest -- an attempt at that measured the
# similarity of unrelated passages and produced 0.87, higher than any question
# reaches, because two passages of one chemistry book resemble each other far
# more than a short question resembles either. It needs questions to calibrate
# against, and packing has none.
NOTHING_NEAR = 0.635
STANDS_CLEAR = 0.25

# Both were left where they are on purpose, and the reason is worth keeping
# because the obvious repair was tried and does not survive its own evidence.
#
# Measured over eighteen questions on a corpus of five statistics books, this
# threshold is above every clearance that corpus produces -- the range there is
# 0.0000 to 0.1515 -- so on that corpus the second condition holds for all
# eighteen and the warning is decided by the first signal alone. Read from
# there, 0.25 looks like a threshold measuring nothing, and lowering it into
# that range looks free.
#
# It is not. On the corpus this constant was measured on, an unrelated question
# clears 0.172 and an answered one clears 0.331; the two ranges genuinely
# overlap, and 0.14 would let that unrelated question through while rescuing
# only one of the two answered questions the statistics corpus marks wrongly.
# There is no value that separates both banks, which is the finding, not a
# number to move.
#
# NOTHING_NEAR has the same problem from the other side: on the statistics
# corpus it sits inside the answered range rather than below it, and the band
# that would fix that bank -- 0.4978 to 0.6080 -- belongs to that package.
# Moving it has been tried twice on evidence that looked as good.
#
# What the two banks together say is that the shape is wrong rather than the
# values, which is what the comment above already suspected.

# Which passage stands in for the tail. Far enough down that a handful of real
# answers do not drag it up, near enough that it is still the same corpus.
TAIL_AT = 50


# What share of its own reach a corpus has to come, for the question to count as
# one it is about. Measured over four packages built for this: the worst
# question the corpus answers reaches 0.68 of its reach and the best unrelated
# one 0.53, so the cut goes between them.
#
# This is still a constant, but not the same kind. It is a constant about how
# questions relate to corpora, applied to a number each corpus measured about
# itself -- which is what the absolute threshold could never be, and why that
# one landed inside the answered range of one corpus and below another's.
ANSWERS_AT_SHARE = 0.60

# The same idea for a package that was given its questions, where the reach is
# already the threshold rather than an estimate of one: the lowest any of the
# declared questions reached.
#
# Not 1.0, which is where the arithmetic points and where it fails. Setting the
# cut exactly at the worst declared question marks that very question, because
# the reach is stored rounded to four places and the cosine computed at query
# time falls a little either side of it. This is the width of that rounding and
# of the fine variation around it, and no more: the cut stays against what was
# declared rather than drifting below it.
ASKED_MARGIN = 0.95


def _calibrated_from_questions() -> bool:
    """Whether the reach was measured against real questions rather than probes.

    It changes what the number means. Estimated from passages it is how near a
    question that the corpus answers would come, and the share above turns that
    into a threshold. Taken from the questions themselves it already is the
    threshold -- the lowest any of them reached -- and scaling it again would
    put the cut well below anything measured.
    """
    try:
        packages = _open_packages()
    except Exception:  # noqa: BLE001 - no package open is not "from focus"
        return False
    for package in packages:
        try:
            row = package["connection"].execute(
                "SELECT value FROM meta WHERE key='answerable_at_from'").fetchone()
        except Exception:  # noqa: BLE001
            continue
        if row and json.loads(row[0]) == "focus":
            return True
    return False


def _reach_of_open_packages() -> float | None:
    """What the open packages measured as their own reach, or None if none did.

    The smallest, when several are open: a question is about the corpus if it
    is about any of them, and holding it to the reach of the widest would
    condemn a question the narrow one answers.
    """
    reaches = []
    for package in _open_packages():
        value = archive.answerable_at(package["connection"])
        if value:
            reaches.append(value)
    return min(reaches) if reaches else None


def _nothing_near(closeness: float, clearance: float,
                  reach: float | None,
                  from_questions: bool = False) -> bool:
    """Whether to say the corpus is not about the question.

    With a reach measured at packing time the question is judged against what
    this corpus can actually reach. Without one -- a package built before that
    was measured -- the two constants above decide, exactly as they did, so an
    existing package answers as it always has.
    """
    if reach:
        return closeness < reach * (ASKED_MARGIN if from_questions
                                    else ANSWERS_AT_SHARE)
    return closeness < NOTHING_NEAR and clearance < STANDS_CLEAR


def _closest_to(query: str) -> tuple[float, float] | None:
    """How near the corpus comes to a question, and how far that stands clear.

    Returns the best cosine and its distance from the tail of the ranking, or
    None where there is no such number: a package that indexes words alone has
    none, and inventing one from BM25 would be the mistake this replaced.
    """
    packages = _open_packages()
    if not all(archive._semantic_ready(p["connection"]) for p in packages):
        return None
    try:
        import numpy as np

        from . import semantic

        vector = np.asarray(semantic.encode([query], role="query")[0],
                            dtype=np.float32)
        todos: list[float] = []
        for package in packages:
            identifiers, matrix = archive._vectors(package["connection"])
            if identifiers:
                todos.extend((matrix @ vector).tolist())
    except Exception:  # noqa: BLE001
        return None
    if not todos:
        return None
    todos.sort(reverse=True)
    queue = todos[min(TAIL_AT, len(todos) - 1)]
    return float(todos[0]), float(todos[0] - queue)


def _best_similarity(connection, vector) -> float | None:
    """How close this package comes to the query, at its closest passage.

    Unlike a BM25 score, this number means the same thing in every package: it
    is the angle between the query and a passage, measured by one model that
    knows nothing of the corpus the passage was drawn from.
    """
    identifiers, matrix = archive._vectors(connection)
    if not identifiers:
        return None
    return float((matrix @ vector).max())


def _packages_worth_asking(packages: list[dict], query: str) -> list[dict]:
    """Drop the packages that have nothing to say about this query.

    Reciprocal rank compares positions, and a position only means something
    among lists that are about the same thing. Merging one list per package
    assumes every package is equally worth reading, so the first passage of a
    package on an unrelated subject arrives with the weight of the first
    passage of the package that answers -- and the merged list converges on
    equal shares, one in N of it useful.

    The signal that separates them is the one reciprocal rank discards, and
    between packages it is comparable: the cosine of the dense branch. So the
    packages are read first for how near they come, and those far below the
    nearest do not reach the merge at all.

    A package without vectors leaves nothing to compare, and then every package
    is asked, as before: a query that cannot be measured is not one to filter on.
    """
    if len(packages) < 2:
        return packages
    if not all(archive._semantic_ready(p["connection"]) for p in packages):
        return packages

    import numpy as np

    from . import semantic

    vector = np.asarray(semantic.encode([query], role="query")[0], dtype=np.float32)
    closeness = {}
    for package in packages:
        best = _best_similarity(package["connection"], vector)
        if best is None:
            return packages
        closeness[package["name"]] = best

    cap = max(closeness.values())
    return [p for p in packages if closeness[p["name"]] >= cap - RELEVANCE_MARGIN]


# What share of a query's terms the index must hold for the lexical ranking to
# be worth merging. Measured on questions asked in and out of the language of a
# corpus: the ones in it had three quarters or more of their terms indexed, the
# ones outside it a quarter or fewer, and nothing landed between.
LEXICAL_FOOTHOLD = 0.5


def _mode_for(package: dict, query: str) -> str:
    """Which engines can say anything useful about this query, in this package.

    Retrieval by word and retrieval by meaning are merged by rank, which assumes
    both lists carry information. When the query is written in a language the
    package is not, the lexical half cannot match anything real: what it ranks
    are accidents -- a surname, an abbreviation, a number that happens to appear
    -- and there are few of them, so they sit at the top of their own list. Rank
    fusion rewards exactly that, and a bibliographic reference about
    Staphylococcus aureus arrives first for a question about oxygen in blood.

    The damage is concentrated where it costs most. Measured over five pairs of
    equivalent questions, the passages that answer still come back inside the
    top five either way -- the average similarity differs by seven hundredths --
    but at the first position, which is what gets cited, the gap is twenty-six.

    So a query whose words are mostly absent from the package is answered by
    meaning alone.

    Two signals, and both are required, because each one alone is wrong in a
    way the other is not.

    How much of the query the index holds depends on how much is in the package.
    On a corpus of three hundred pages a question in its own language has three
    quarters of its terms indexed and one in another language a quarter; on a
    package of four short documents, a perfectly ordinary English question has
    a third, because most of the words are simply not in there yet.

    What language the query looks like misreads short questions: "how does a
    catalyst work" is reported as Portuguese with the same confidence a real
    Spanish question is reported as Spanish, so no threshold on that separates
    them either.

    Together they do. A question in the language of the package is spared by the
    second signal however little of it is indexed; a question in the package's
    language whose words are simply rare is spared by the first.

    The function next door that reports a mismatch only when *no* term appears
    at all is a third test, and it is backwards for the case that hurts: a query
    matching nothing produces no lexical ranking, so there is nothing to fuse
    and nothing goes wrong. The damage comes from the query that matches a few
    words by accident -- "reaccion acido base" finds eleven passages in an
    English corpus because "base" is an English word too, and those eleven
    arrive first.
    """
    try:
        from . import search as _search

        terms = set(_search.searchable_terms(_search._normalize(query)))
        terms -= _search.STOPWORDS
        if not terms:
            return "auto"
        markers = ",".join("?" * len(terms))
        with archive._CONNECTION_LOCK:
            row = package["connection"].execute(
                f"SELECT count(*) FROM df WHERE term IN ({markers})",
                tuple(terms)).fetchone()
        present = row[0] if row else 0
        if present / len(terms) >= LEXICAL_FOOTHOLD:
            return "auto"

        corpus_language = archive._corpus_language(package["connection"])
        language_of_query, _ = _search.detect_language(query)
        if corpus_language and language_of_query and language_of_query != corpus_language:
            return "semantic"
    except Exception:  # noqa: BLE001
        pass
    return "auto"


def search_packages(query: str, limit: int = 5,
                    only: str | None = None) -> list[dict]:
    """Search every configured package and return one ranked list.

    Scores from different packages are computed over different corpus
    statistics -- the frequency of a term depends on the corpus it is measured
    in -- so they cannot be compared to one another. Position within each
    package can, which is what reciprocal rank merges.

    Which packages take part is decided first: merging positions across
    packages that are not about the same thing buries the answer, so a package
    far from the query never reaches the merge.
    """
    configured = _open_packages()
    if len(configured) == 1:
        return archive.query(configured[0]["connection"], query, limit=limit,
                             only=only, mode=_mode_for(configured[0], query))

    packages = _packages_worth_asking(configured, query)

    def labelled(package):
        """Results carry the package they came from, since several are served."""
        for item in archive.query(package["connection"], query, limit=limit,
                                  only=only, mode=_mode_for(package, query)):
            item = dict(item)
            item["package"] = package["name"]
            yield item

    if len(packages) == 1:
        return list(labelled(packages[0]))[:limit]

    from .semantic import fuse

    by_key: dict = {}
    item_lists: list[list] = []
    for package in packages:
        items = []
        for item in labelled(package):
            key = (package["name"], item["document"], item["passage"][:120])
            by_key[key] = item
            items.append(key)
        item_lists.append(items)
    return [by_key[c] for c in fuse(item_lists)[:limit]]


def create_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="mdcx",
        title="Queryable document corpus",
        instructions=(
            "Queries a converted and verified document collection. Use search to "
            "locate the passages that answer a question: it returns the verbatim "
            "text together with the source of each passage, so the source can be "
            "cited rather than recalled. Use info to learn what the corpus "
            "contains before querying it."
        ),
    )

    @server.tool(
        name="search",
        title="Search the corpus",
        description=(
            "Returns the passages that answer a query, each with its source "
            "document, portable path, whether it was sent or received, and its "
            "rank. The list is ordered by that rank and by nothing else: the "
            "two engines behind it score on scales with no common meaning, so "
            "there is no number here to sort or filter by. "
            "The reply carries `similarity`, how near the corpus comes to the "
            "question, and a `warning` when nothing in it is about the question "
            "-- the passages are still returned, being the nearest there are. "
            "When the package was built with meaning indexed, a query reaches "
            "documents written in other languages as well; use info to see "
            "whether it was. Without that, matching is by word and the query "
            "should be written in the language of the documents. "
            "`limit` is how many passages to return, from 1 to 20; a value "
            "outside that range is brought within it rather than refused. "
            "Set `direction` to `received` or `sent` to restrict the search to "
            "one side of a correspondence; omit it to search all of them. Any "
            "other value is treated as omitted, so a search is never narrowed "
            "by a word this tool does not recognise."
        ),
    )
    async def search(query: str, limit: int = 5,
                     direction: str | None = None) -> dict:
        """Return passages answering the query.

        query: the question, phrased as it would be asked of a person.
        limit: number of passages to return, between 1 and 20.
        direction: "received" or "sent" to restrict the search; omit for all.
        """
        top = max(1, min(int(limit), 20))
        scope = (direction or "").lower().strip() or None
        if scope not in ("received", "sent"):
            scope = None
        results = search_packages(query, top, scope)
        # What comes back is a position, not a measurement. The two engines
        # score on scales with nothing in common -- BM25 has no upper bound and
        # depends on the corpus it was measured in, cosine runs from zero to one
        # and does not -- and the list is ordered by neither: it is merged by
        # rank, because that is the only thing the two agree on. Publishing a
        # `score` beside the passage invited comparing, sorting and filtering by
        # it, and none of the three was ever valid. A query the corpus cannot
        # answer reached 14.65 while one it answers well sat at 0.65.
        answer = {
            "query": query,
            "found": len(results),
            "passages": [
                {
                    "document": item["document"],
                    "direction": item["source"].lower(),
                    "path": item["pseudopath"],
                    "rank": position,
                    "text": item["passage"],
                    **({"package": item["package"]} if "package" in item else {}),
                }
                for position, item in enumerate(results, start=1)
            ],
        }
        measured = _closest_to(query)
        if measured is not None:
            closeness, clearance = measured
            answer["similarity"] = round(closeness, 4)
            answer["stands_clear"] = round(clearance, 4)
            reach = _reach_of_open_packages()
            if reach is not None:
                answer["answerable_at"] = round(reach, 4)
            if _nothing_near(closeness, clearance, reach,
                             _calibrated_from_questions()):
                answer["warning"] = (
                    "nothing in this corpus is about the question: the best "
                    "passage is no nearer than the rest, so the passages below "
                    "are the nearest there are rather than an answer")
        if not results:
            warning = archive.language_mismatch(connection, query)
            if warning:
                answer["hint"] = warning
        return answer

    def _cross_language() -> str:
        """How this server can answer, in the terms that change the answer."""
        connection = _connection()
        if not archive.has_vectors(connection):
            return "no: this package indexes words only"
        if not archive._semantic_ready(connection):
            return ("not in this installation: the package indexes meaning, but "
                    "the multilingual extra is missing or set to another model")
        return "yes: a query in one language reaches documents in the others"

    def _describe(package: dict) -> dict:
        header = package["header"]
        return {
            "package": package["name"],
            "format": f"{header.get('file_format')} v{header.get('version')}",
            "issuer": header.get("issuer") or "(not declared)",
            "created_utc": header.get("created_utc"),
            "documents": header.get("documents"),
            "passages": header.get("passages"),
            "language": header.get("language") or "(not detected)",
            "integrity": "intact" if header.get("_intact") else "ALTERED",
            "conversion": header.get("conversion", {}),
        }

    @server.tool(
        name="info",
        title="Corpus information",
        description=(
            "Describes the package: number of documents, creation date, issuer "
            "and the fidelity with which it was converted from the originals."
        ),
    )
    async def info() -> dict:
        """Return the corpus record without querying it."""
        packages = _open_packages()
        if len(packages) == 1:
            description = _describe(packages[0])
            description.pop("package")
            description["cross_language_search"] = _cross_language()
            return description

        # Several packages are queried as one corpus, so the totals describe the
        # whole of it and the list says where each part comes from.
        parts = [_describe(p) for p in packages]
        return {
            "packages": len(parts),
            "documents": sum(p["documents"] or 0 for p in parts),
            "passages": sum(p["passages"] or 0 for p in parts),
            "integrity": ("intact" if all(p["integrity"] == "intact" for p in parts)
                          else "ALTERED"),
            "cross_language_search": _cross_language(),
            "each": parts,
        }

    @server.tool(
        name="document",
        title="Read a full document",
        description=(
            "Returns the complete text of one document, identified by name or by "
            "portable path. Use it only when passages are insufficient: a full "
            "document may span tens of thousands of tokens."
        ),
    )
    async def document(name: str) -> dict:
        """Return the full text of one document in the corpus."""
        packages = _open_packages()
        for package in packages:
            connection = package["connection"]
            row = connection.execute(
                "SELECT d.name, d.pseudopath, d.source, "
                "       group_concat(p.text, char(10) || char(10)) "
                "FROM document d JOIN passage p ON p."
                + archive.document_column(connection) + " = d.id "
                "WHERE d.name = ? OR d.pseudopath = ? "
                "GROUP BY d.id ORDER BY p.position LIMIT 1",
                (name, name)).fetchone()
            if row:
                output = {"found": True, "document": row[0], "path": row[1],
                          "direction": row[2], "text": row[3] or ""}
                if len(packages) > 1:
                    output["package"] = package["name"]
                return output
        where = "this corpus" if len(packages) == 1 else f"any of the {len(packages)} packages"
        return {"found": False,
                "message": f"No document named {name!r} in {where}."}

    return server


def main() -> int:
    try:
        _connection()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot open corpus: {exc}", file=sys.stderr)
        return 2

    try:
        server = create_server()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot start server: {exc}", file=sys.stderr)
        return 2

    header = _STATE.get("header", {})
    print(f"mdcx: {header.get('documents')} documents ready.", file=sys.stderr)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
