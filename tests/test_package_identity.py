"""Checks that a cache belongs to a package and not to a memory address.

Three caches here are filled once per package and never invalidated, which is
correct: a package is immutable once written. They were keyed by
`id(connection)`. CPython reuses an address as soon as the object at it is
collected, so a package opened after another was closed could land on the same
address and read what the closed one had left.

Nothing raised. The vector cache is the one that mattered: passage text is
fetched from the live connection by passage id, so the document names and the
pseudopaths on the reply were the right package's. Only the ranking and the
cosine came from a different corpus -- and a reply carries no evidence of which
corpus ranked it. Measured on two packages of two documents each, the second
answered a botany question with a cosine of 0.610849, which was the first
package's score, in place of its own 0.409538.

It showed up as an intermittent failure in test_relevance, where the gate that
decides which packages are worth asking compares those cosines: with one
package's distances read from another's vectors, the gate admitted too many or
too few, and the two tests that assert opposite sides of it failed in turn,
about one run in three, never when run alone.

The key is now the digest of the decrypted body, which open_package records.
Equal digest means equal content, so two connections onto the same package share
a cache entry and two connections onto different packages cannot.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, semantic  # noqa: E402

needs_model = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")

QUERY = "how does a plant make its own food"


def _build(root: Path, name: str, documents: dict[str, str],
               with_semantics: bool) -> Path:
    folder = root / name / "Received"
    folder.mkdir(parents=True)
    for title, text in documents.items():
        (folder / f"{title}.md").write_text(
            f"---\nsource_format: pdf\n---\n\n# {title}\n\n{text}\n",
            encoding="utf-8")
    target = root / f"{name}.mdcx"
    archive.pack(root / name, target, "k", semantic=with_semantics)
    return target


BOTANICA = {
    "photosynthesis": "Plants convert sunlight into sugar using chlorophyll.",
    "roots": "Roots draw water and minerals from the soil into the stem.",
}
IMPRENTA = {
    "gutenberg": "Gutenberg built the printing press with movable type.",
    "typography": "A typeface is cut in metal and inked before pressing paper.",
}


def _keys_are_digests(name: str, cache: dict) -> None:
    assert cache, f"{name} is empty, so this checked nothing"
    for key in cache:
        assert isinstance(key, str), (
            f"{name} holds {key!r}, which looks like an address; "
            f"an address is reused and the entry under it is not")


def test_the_lexical_caches_are_keyed_by_content_and_not_by_address(tmp_path):
    """An address is not an identity, and these caches outlive the object at it."""
    package = _build(tmp_path, "botanica", BOTANICA, with_semantics=False)
    connection, _ = archive.open_package(package, "k")
    archive.query(connection, "sunlight sugar chlorophyll", limit=2)
    _keys_are_digests("_STATS_CACHE", archive._STATS_CACHE)
    _keys_are_digests("_COLUMN_CACHE", archive._COLUMN_CACHE)


@needs_model
def test_the_vector_cache_is_keyed_by_content_and_not_by_address(tmp_path):
    """This is the one that was keyed by address, and the one that mattered.

    It has to be checked on a package that actually holds vectors: on one built
    without them the cache stays empty and an assertion over its keys passes
    without having looked at anything.
    """
    package = _build(tmp_path, "botanica", BOTANICA, with_semantics=True)
    connection, _ = archive.open_package(package, "k")
    archive.semantic_query(connection, QUERY, 1)
    _keys_are_digests("_VECTOR_CACHE", archive._VECTOR_CACHE)


def test_two_connections_onto_one_package_share_the_entry(tmp_path):
    """The fix must not turn the cache off: that is what it is for."""
    package = _build(tmp_path, "botanica", BOTANICA, with_semantics=False)
    first, _ = archive.open_package(package, "k")
    second, _ = archive.open_package(package, "k")
    assert first is not second
    assert archive._cache_key(first) == archive._cache_key(second)
    assert (archive._corpus_statistics(first)
            is archive._corpus_statistics(second))


def test_a_connection_this_module_did_not_open_is_not_cached(tmp_path):
    """With no digest there is no key that could tell it from another."""
    import sqlite3

    stray = sqlite3.connect(":memory:")
    stray.execute("CREATE TABLE df (term TEXT, passages INTEGER)")
    stray.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    assert archive._cache_key(stray) is None
    before = len(archive._STATS_CACHE)
    archive._corpus_statistics(stray)
    assert len(archive._STATS_CACHE) == before, (
        "it was cached under a key that cannot identify it")


@needs_model
def test_a_package_does_not_answer_with_a_closed_one_s_vectors(tmp_path):
    """The failure itself: same address, different package, another's ranking.

    The address has to be recycled for this to be exercised at all, which is up
    to the allocator. In practice it is reused on the first attempt, because the
    connection just released is the best fit for the one being made. If it is
    not, there is nothing here to measure and the test says so rather than
    passing quietly.
    """
    a = _build(tmp_path, "botanica", BOTANICA, with_semantics=True)
    b = _build(tmp_path, "imprenta", IMPRENTA, with_semantics=True)

    conn_a, _ = archive.open_package(a, "k")
    address = id(conn_a)
    archive.semantic_query(conn_a, QUERY, 1)
    del conn_a
    gc.collect()

    for _ in range(200):
        conn_b, _ = archive.open_package(b, "k")
        if id(conn_b) == address:
            break
        del conn_b
        gc.collect()
    else:
        pytest.skip("the allocator never reused the address")

    with_cache = archive.semantic_query(conn_b, QUERY, 1)[0]
    archive._VECTOR_CACHE.clear()
    without_cache = archive.semantic_query(conn_b, QUERY, 1)[0]

    assert with_cache["score"] == pytest.approx(without_cache["score"]), (
        "the package answered with a cosine that is not its own")
    assert with_cache["document"] == without_cache["document"], (
        "the ranking came from another package's vectors")
