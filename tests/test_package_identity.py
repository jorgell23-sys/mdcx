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

necesita_modelo = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")

CONSULTA = "how does a plant make its own food"


def _construir(raiz: Path, nombre: str, documentos: dict[str, str],
               con_significado: bool) -> Path:
    carpeta = raiz / nombre / "Received"
    carpeta.mkdir(parents=True)
    for titulo, texto in documentos.items():
        (carpeta / f"{titulo}.md").write_text(
            f"---\nsource_format: pdf\n---\n\n# {titulo}\n\n{texto}\n",
            encoding="utf-8")
    destino = raiz / f"{nombre}.mdcx"
    archive.pack(raiz / nombre, destino, "k", semantic=con_significado)
    return destino


BOTANICA = {
    "photosynthesis": "Plants convert sunlight into sugar using chlorophyll.",
    "roots": "Roots draw water and minerals from the soil into the stem.",
}
IMPRENTA = {
    "gutenberg": "Gutenberg built the printing press with movable type.",
    "typography": "A typeface is cut in metal and inked before pressing paper.",
}


def _claves_son_digests(nombre: str, cache: dict) -> None:
    assert cache, f"{nombre} is empty, so this checked nothing"
    for clave in cache:
        assert isinstance(clave, str), (
            f"{nombre} holds {clave!r}, which looks like an address; "
            f"an address is reused and the entry under it is not")


def test_the_lexical_caches_are_keyed_by_content_and_not_by_address(tmp_path):
    """An address is not an identity, and these caches outlive the object at it."""
    paquete = _construir(tmp_path, "botanica", BOTANICA, con_significado=False)
    conexion, _ = archive.open_package(paquete, "k")
    archive.query(conexion, "sunlight sugar chlorophyll", limit=2)
    _claves_son_digests("_STATS_CACHE", archive._STATS_CACHE)
    _claves_son_digests("_COLUMN_CACHE", archive._COLUMN_CACHE)


@necesita_modelo
def test_the_vector_cache_is_keyed_by_content_and_not_by_address(tmp_path):
    """This is the one that was keyed by address, and the one that mattered.

    It has to be checked on a package that actually holds vectors: on one built
    without them the cache stays empty and an assertion over its keys passes
    without having looked at anything.
    """
    paquete = _construir(tmp_path, "botanica", BOTANICA, con_significado=True)
    conexion, _ = archive.open_package(paquete, "k")
    archive.semantic_query(conexion, CONSULTA, 1)
    _claves_son_digests("_VECTOR_CACHE", archive._VECTOR_CACHE)


def test_two_connections_onto_one_package_share_the_entry(tmp_path):
    """The fix must not turn the cache off: that is what it is for."""
    paquete = _construir(tmp_path, "botanica", BOTANICA, con_significado=False)
    primera, _ = archive.open_package(paquete, "k")
    segunda, _ = archive.open_package(paquete, "k")
    assert primera is not segunda
    assert archive._cache_key(primera) == archive._cache_key(segunda)
    assert (archive._corpus_statistics(primera)
            is archive._corpus_statistics(segunda))


def test_a_connection_this_module_did_not_open_is_not_cached(tmp_path):
    """With no digest there is no key that could tell it from another."""
    import sqlite3

    suelta = sqlite3.connect(":memory:")
    suelta.execute("CREATE TABLE df (term TEXT, passages INTEGER)")
    suelta.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    assert archive._cache_key(suelta) is None
    antes = len(archive._STATS_CACHE)
    archive._corpus_statistics(suelta)
    assert len(archive._STATS_CACHE) == antes, (
        "it was cached under a key that cannot identify it")


@necesita_modelo
def test_a_package_does_not_answer_with_a_closed_one_s_vectors(tmp_path):
    """The failure itself: same address, different package, another's ranking.

    The address has to be recycled for this to be exercised at all, which is up
    to the allocator. In practice it is reused on the first attempt, because the
    connection just released is the best fit for the one being made. If it is
    not, there is nothing here to measure and the test says so rather than
    passing quietly.
    """
    a = _construir(tmp_path, "botanica", BOTANICA, con_significado=True)
    b = _construir(tmp_path, "imprenta", IMPRENTA, con_significado=True)

    con_a, _ = archive.open_package(a, "k")
    direccion = id(con_a)
    archive.semantic_query(con_a, CONSULTA, 1)
    del con_a
    gc.collect()

    for _ in range(200):
        con_b, _ = archive.open_package(b, "k")
        if id(con_b) == direccion:
            break
        del con_b
        gc.collect()
    else:
        pytest.skip("the allocator never reused the address")

    con_cache = archive.semantic_query(con_b, CONSULTA, 1)[0]
    archive._VECTOR_CACHE.clear()
    sin_cache = archive.semantic_query(con_b, CONSULTA, 1)[0]

    assert con_cache["score"] == pytest.approx(sin_cache["score"]), (
        "the package answered with a cosine that is not its own")
    assert con_cache["document"] == sin_cache["document"], (
        "the ranking came from another package's vectors")
