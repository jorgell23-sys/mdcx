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

"""Keeping the expressions that word matching throws away.

A lexical index is built out of words, and the rule that decides what a word is
discards the symbols. For prose that is right: nobody searches for a comma. For
a corpus of mathematics it removes the content.

Measured on a real corpus of 25.1 million passages, by a consumer who ran into
it:

    pq | b(b+p+q)   -- true
    pq | b(b-p-q)   -- false, one sign apart

Both retrieved the same 400 passages and the same first document, because
`searchable_terms` returns the same set for the two of them. The statement and
its negation are one query. Asked on their own, without prose around them, they
return no terms at all and retrieve nothing whatever -- the words are not
merely insufficient, there are none.

What that cost downstream is worth stating, because it is the shape of the
damage: a consumer built a novelty check on top of "does the corpus already say
this", and had to take its power away again -- it can say "this is about that"
and not "this was already known".

So the expressions are kept beside the words, as their own index, and only for
a corpus that asks for one. Extraction is deliberately narrow: a token that
mixes symbols with alphanumerics. `f(x)` and `b(b+p+q)` are expressions;
`hello,` and `--force` are not, and neither is prose.
"""
from __future__ import annotations

import sqlite3
import unicodedata

# What makes a token an expression rather than a word. An operator or a bracket
# is enough, provided something is being operated on: a run of punctuation on
# its own carries nothing to search for.
OPERATORS = set("+-*/^|=<>()[]{}\\_~")

# Below this a token says too little to be worth an index entry. `f(x)` is four
# characters and is the shortest thing anyone would look for.
MINIMUM_LENGTH = 4

# What is stripped from the ends before an expression is judged. Sentence
# punctuation belongs to the sentence, not to the formula it follows.
EDGES = ".,;:!?\"'«»“”‘’"

EXPRESSION_VERSION = 1


def normalise(expression: str) -> str:
    """One spelling for expressions that differ only in how they were typed.

    The minus sign is the case that matters: a document may carry U+2212 where
    a question is typed with a hyphen, and they are the same statement. Case is
    folded because a corpus is not consistent about it either.
    """
    text = unicodedata.normalize("NFKC", expression)
    for dash in "−–—‐‑":
        text = text.replace(dash, "-")
    return text.strip(EDGES).lower()


def extract(text: str) -> list[str]:
    """The expressions in this text, normalised, in the order they appear.

    Split on whitespace rather than parsed. An expression that carries a space
    is not recovered, which is a limit and a deliberate one: deciding where a
    formula ends inside a sentence is a different problem, and guessing at it
    would fill the index with fragments of prose.
    """
    found: list[str] = []
    for token in text.split():
        candidate = normalise(token)
        if len(candidate) < MINIMUM_LENGTH:
            continue
        if not any(c in OPERATORS for c in candidate):
            continue
        if not any(c.isalnum() for c in candidate):
            continue
        found.append(candidate)
    return found


def build(connection: sqlite3.Connection) -> dict:
    """Index the expressions of every passage. Returns what was found.

    One row per expression per passage, which is what lets a question ask
    "which passages state this" rather than "which passages are about this".
    """
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS expression (
            expr TEXT NOT NULL,
            passage_id INTEGER NOT NULL REFERENCES passage(id)
        );
        CREATE INDEX IF NOT EXISTS expression_by_expr ON expression(expr);
    """)
    rows = connection.execute("SELECT id, text FROM passage").fetchall()
    entries: list[tuple[str, int]] = []
    for identifier, text in rows:
        for expression in dict.fromkeys(extract(text)):
            entries.append((expression, identifier))
    connection.executemany("INSERT INTO expression VALUES (?,?)", entries)
    distinct = connection.execute(
        "SELECT count(DISTINCT expr) FROM expression").fetchone()[0]
    return {"expression_version": EXPRESSION_VERSION,
            "expressions": distinct,
            "expression_entries": len(entries)}


def has_index(connection: sqlite3.Connection) -> bool:
    """Whether this package carries an expression index.

    The version is checked as well as the table: a package written by a later
    rule would be read under this one, and silently answer about a different
    spelling.
    """
    try:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='expression'").fetchone()
        if not present:
            return False
        row = connection.execute(
            "SELECT value FROM meta WHERE key='expression_version'").fetchone()
    except sqlite3.Error:
        return False
    if row is None:
        return False
    try:
        import json

        return int(json.loads(row[0])) == EXPRESSION_VERSION
    except (ValueError, TypeError):
        return False


def passages_stating(connection: sqlite3.Connection,
                     expressions: list[str]) -> dict[str, list[int]]:
    """Which passages carry each of these expressions, by exact match.

    Exact on purpose. The whole reason this index exists is that a statement
    and its negation differ by one character, so anything that treats them as
    near enough returns the defect it was built to remove.
    """
    found: dict[str, list[int]] = {}
    for expression in dict.fromkeys(expressions):
        rows = connection.execute(
            "SELECT passage_id FROM expression WHERE expr = ?",
            (expression,)).fetchall()
        if rows:
            found[expression] = [r[0] for r in rows]
    return found
