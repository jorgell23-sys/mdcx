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

"""Recovering terms the transcription got wrong, by the shape of their letters.

A word that came out of optical recognition with one letter misread is a word
the index does not contain, and literal matching can never return it. The
passage is in the corpus and the question cannot reach it. There is no error to
show for it either: the reply is simply empty.

Optical recognition does not misread letters at random. It misreads the ones
that look alike printed -- 0 for O, 1 for l, 5 for S -- so grouping letters by
how they are drawn puts a misread word and its original in the same bucket,
which is what makes them findable again.

The shape alone is far too coarse to answer with: one bucket holds hundreds of
words. So it never answers. It is the cheap sieve that proposes a few
candidates, and each candidate is then verified against the query by edit
distance. Measured over 2,976 words of one corpus corrupted with documented
confusions, that pairing recovers 76.5 per cent of them against 0 per cent for
literal matching and 45.3 per cent for trigram overlap, at a seventh of the
trigram cost.

The technique is not new and the attribution belongs here: it is the character
shape codes Spitz described at Xerox in the nineties, used there over document
images to avoid a full recognition pass. What is different is where it is
applied -- over text already converted, for the errors the conversion left
behind.

It is worth stating what this does not do, because the method invites the
larger claim. It is not a spell checker: the same measurement over randomly
substituted letters, the error a typist makes, falls to 49.6 per cent while
trigrams do not move. The advantage comes from the shape, which is also its
boundary.

And it is not precise. Half of what it proposes is not a transcription error --
49.9 per cent against 59.1 for trigrams -- because for Latin script the table is
coarse: all ten digits and 22 of the 26 lowercase letters fall in one class, so
what the sieve really proposes is any single substitution among them. `libre`
finds `libro`. That is the price of a filter cheap enough to run over a whole
vocabulary, and it is bearable only because the reply says which word was read
as which. A caller shown the passage alone would have no way to tell.

One thing the proposal carried that is not here: a function to rewrite a whole
query string with its widened terms. `lexical_query` builds its own expression
for the word index, quoted and segmented the way that index needs, so a second
way of writing the same thing would have been an untested path that produced
subtly different queries. What a caller needs from the widening is the report of
what was read as what, and that travels in `notes`.

Nothing here is required. A corpus without the index answers exactly as it does
today: every entry point checks first, and a package built before this returns
nothing from it rather than failing.
"""
from __future__ import annotations

import sqlite3

from . import search as _search

# The mapping is versioned and travels in the package, so an index built under
# one table is never read under another. The keys would not line up, and the
# failure would be silent -- returning nothing rather than returning wrong,
# which is the harder kind to notice.
SHAPE_VERSION = 1

# Each character mapped onto the class of its printed shape: how many groups of
# dots touch the top of the glyph, and how many touch the bottom. Derived by
# rasterising real typefaces on a 9x7 grid, the order of the matrices dot
# printers used, and frozen here -- the derivation needs a font file, an imaging
# library and a rasteriser, none of which belong in a package that has to
# install anywhere.
#
#   a  one group above, one below   b c d e f g i j l o p q r s t z 0..9
#   b  one above, two below         a h k
#   c  one above, three below       m
#   d  two above, one below         v y
#   e  two above, two below         n u x
#   f  three above, two below       w
#
# Only lowercase letters and digits appear, because every entry point normalises
# first: the index is built and read through the package's own tokeniser, which
# folds case and strips diacritics.
TABLE = {
    "0": "a", "1": "a", "2": "a", "3": "a", "4": "a", "5": "a", "6": "a", "7": "a",
    "8": "a", "9": "a", "a": "b", "b": "a", "c": "a", "d": "a", "e": "a", "f": "a",
    "g": "a", "h": "b", "i": "a", "j": "a", "k": "b", "l": "a", "m": "c", "n": "e",
    "o": "a", "p": "a", "q": "a", "r": "a", "s": "a", "t": "a", "u": "e", "v": "d",
    "w": "f", "x": "e", "y": "d", "z": "a",
}

# Below this length a shape key says almost nothing: a three-letter word shares
# its key with hundreds of others, so the sieve would propose noise and the
# verification would spend its time refusing it.
MINIMUM_LENGTH = 5


def shape_key(term: str) -> str:
    """The shape of a term: one class per character, in order.

    A character the table does not know keeps itself, so a term mixing scripts
    produces a stable key of its own instead of collapsing into its neighbours.
    """
    return "".join(TABLE.get(c, c) for c in term)


def within_one_edit(a: str, b: str) -> bool:
    """Whether one insertion, deletion or substitution turns one into the other.

    Written out rather than taken from a library, because it runs on every
    candidate the sieve proposes and this package has no dependency to take it
    from.

    One edit and no more, which is what bounds the sieve to the errors it was
    measured on: a single misread letter. Two letters read as one -- `rn` for
    `m`, the other confusion optical recognition makes -- is outside this, and
    outside the shape table too, where `rn` and `m` do not share a key.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        differences = 0
        for x, y in zip(a, b):
            if x != y:
                differences += 1
                if differences > 1:
                    return False
        return differences == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def has_index(connection: sqlite3.Connection) -> bool:
    """Whether this package carries a shape index this code can read.

    A package written before this does not, and has to keep answering: every
    entry point asks here first and falls back to what it did before.

    The version is checked and not merely written. An index built under one
    table and read under another lines up on nothing, and the failure is the
    quiet kind -- an empty answer where there was one, with no error to show
    for it. A package whose version this code does not know is treated as
    having no index, which is what it effectively has.
    """
    try:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='term_shape'"
        ).fetchone()
        if row is None:
            return False
        stored = connection.execute(
            "SELECT value FROM meta WHERE key = 'shape_version'").fetchone()
    except sqlite3.Error:
        return False
    if stored is None:
        return False
    try:
        return int(str(stored[0]).strip().strip('"')) == SHAPE_VERSION
    except (TypeError, ValueError):
        return False


def build(connection: sqlite3.Connection) -> dict:
    """Write the shape index from the terms the package already lists.

    The vocabulary comes from `df`, which packing fills with every term in the
    corpus. Deriving the index from that rather than from the passages keeps the
    two vocabularies identical: an index over a different set of terms would
    propose candidates the search engine then cannot find, which reads from
    outside like the sieve failing rather than like the two disagreeing.
    """
    connection.execute("""
        CREATE TABLE IF NOT EXISTS term_shape (
            term  TEXT NOT NULL,
            shape TEXT NOT NULL
        )""")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS term_shape_by_shape ON term_shape (shape)")
    connection.execute("DELETE FROM term_shape")

    rows = [(term, shape_key(term))
            for (term,) in connection.execute("SELECT term FROM df")
            if term and len(term) >= MINIMUM_LENGTH]
    connection.executemany(
        "INSERT INTO term_shape (term, shape) VALUES (?, ?)", rows)
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('shape_version', ?)",
        (str(SHAPE_VERSION),))
    connection.commit()
    return {"shape_terms": len(rows)}


def candidates(connection: sqlite3.Connection, term: str,
               limit: int = 8) -> list[str]:
    """Terms of the corpus that could be what this one was before transcription.

    The shape narrows the corpus to the words drawn the same way; the edit
    distance decides among them. Neither is enough alone -- the first is too
    coarse to answer with, the second too slow to run over a whole vocabulary --
    and that is the whole design.
    """
    if not term or len(term) < MINIMUM_LENGTH or not has_index(connection):
        return []
    found: list[str] = []
    for (other,) in connection.execute(
            "SELECT term FROM term_shape WHERE shape = ?", (shape_key(term),)):
        if other != term and within_one_edit(term, other):
            found.append(other)
            if len(found) >= limit:
                break
    return found


def expand(connection: sqlite3.Connection, terms) -> dict:
    """For each query term the corpus lacks, the terms it might have been.

    Only absent terms are widened. A term the corpus does contain is answered by
    the index as it stands, and widening it would trade an exact match for
    candidates nobody asked for.

    The terms are normalised here rather than assumed to be. `df` holds folded
    keys, so an unfolded term would be absent from it for the wrong reason --
    reported as unknown, then given a shape key over characters the table does
    not hold, which matches nothing. The whole thing would do nothing at all,
    quietly.
    """
    if not has_index(connection):
        return {}
    out: dict[str, list[str]] = {}
    for raw in terms:
        term = _search._normalize(raw or "")
        if len(term) < MINIMUM_LENGTH or term in out:
            continue
        if connection.execute(
                "SELECT 1 FROM df WHERE term = ?", (term,)).fetchone() is not None:
            continue
        found = candidates(connection, term)
        if found:
            out[term] = found
    return out
