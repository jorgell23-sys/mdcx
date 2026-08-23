"""Checks the restriction of a search to one side of a correspondence.

A collection of correspondence has a direction, and a question is usually about
one side of it. The direction is taken from the top-level folder and stored with
each document, and a query can be restricted to it.

The restriction was written twice and the two halves did not agree on the form
of the value. The stored form is uppercase; the callers all produce lowercase --
the CLI declares its choices that way and the server lowercases what it is
given. The lexical engine folded the case before comparing and the dense one did
not, so a restricted query returned its lexical results and silently dropped
every dense one. Where the query was written in the language of the documents
this looked like nothing at all, because the lexical engine had already found
them; where it was not, the reply came back empty and read as a corpus holding
nothing on the subject.

That is the failure these tests hold shut: not that the filter works, but that
both engines agree on what they are filtering by.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, semantic  # noqa: E402

necesita_modelo = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")


def _build(raiz: Path, *, con_significado: bool) -> Path:
    """A collection with both directions present, in English."""
    recibidos = raiz / "src" / "Received"
    emitidos = raiz / "src" / "Sent"
    recibidos.mkdir(parents=True)
    emitidos.mkdir(parents=True)
    (recibidos / "incoming.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Photosynthesis\n\n"
        "Plants turn sunlight into sugar using the chlorophyll in their "
        "leaves, and release oxygen while doing so.\n", encoding="utf-8")
    (emitidos / "outgoing.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Printing\n\n"
        "Gutenberg built the printing press with movable type in Mainz, "
        "and the page was set one letter at a time.\n", encoding="utf-8")
    paquete = raiz / "corpus.mdcx"
    archive.pack(raiz / "src", paquete, "k", semantic=con_significado)
    return paquete


@pytest.fixture
def lexico(tmp_path):
    conexion, _ = archive.open_package(_build(tmp_path, con_significado=False), "k")
    return conexion


@pytest.fixture
def con_significado(tmp_path):
    conexion, _ = archive.open_package(_build(tmp_path, con_significado=True), "k")
    return conexion


def test_the_direction_comes_from_the_top_level_folder(lexico):
    almacenadas = {fila[0] for fila in
                   lexico.execute("SELECT DISTINCT source FROM document")}
    assert almacenadas == {"RECEIVED", "SENT"}


def test_a_restriction_returns_only_that_side(lexico):
    recibidos = archive.query(lexico, "photosynthesis sunlight", only="received")
    emitidos = archive.query(lexico, "printing press movable type", only="sent")
    assert recibidos and all("incoming" in r["document"] for r in recibidos)
    assert emitidos and all("outgoing" in r["document"] for r in emitidos)


def test_a_restriction_excludes_the_other_side(lexico):
    """The document exists and holds the words; the restriction is what removes it."""
    assert archive.query(lexico, "printing press movable type")
    assert not archive.query(lexico, "printing press movable type", only="received")


def test_the_case_a_caller_writes_does_not_change_the_answer(lexico):
    """Every caller writes lowercase and the column holds uppercase."""
    esperado = archive.query(lexico, "photosynthesis sunlight", only="RECEIVED")
    assert esperado
    for escrito in ("received", "Received", "rEcEiVeD"):
        assert [r["document"] for r in
                archive.query(lexico, "photosynthesis sunlight", only=escrito)] == \
               [r["document"] for r in esperado], f"{escrito!r} answered differently"


def test_omitting_the_direction_reaches_both(lexico):
    documentos = {r["document"] for r in
                  archive.query(lexico, "photosynthesis printing", limit=8)}
    assert len(documentos) == 2


@necesita_modelo
def test_a_restricted_query_still_crosses_languages(con_significado):
    """The regression: the dense engine is the only one that can answer here.

    The query is Spanish and the documents are English, so they share no term
    and the lexical engine contributes nothing. If the restriction drops the
    dense results, what comes back is empty -- which is what happened, and read
    as a corpus with nothing on the subject rather than as a filter fault.
    """
    consulta = "como convierten las plantas la luz solar en azucar"
    assert not archive.query(con_significado, consulta, mode="lexical"), (
        "the premise of this test is that word matching cannot answer it")

    sin_restriccion = archive.query(con_significado, consulta, limit=3)
    assert sin_restriccion, "the dense engine should answer this on its own"

    restringida = archive.query(con_significado, consulta, limit=3, only="received")
    assert restringida, (
        "a restriction to the side that holds the answer returned nothing")
    assert all("incoming" in r["document"] for r in restringida)


@necesita_modelo
def test_a_restriction_still_excludes_the_other_side_by_meaning(con_significado):
    """Reaching across languages must not reach past the restriction."""
    restringida = archive.query(
        con_significado, "quien invento la imprenta de tipos moviles",
        limit=3, only="received")
    assert all("outgoing" not in r["document"] for r in restringida)


def test_the_values_the_cli_offers_are_the_values_that_match(lexico):
    """The choices a user is given have to be choices the column can answer.

    This is the shape of the original fault: two places agreeing on the concept
    and not on the string. The parser is built inside main and cannot be
    inspected without running it, so the declaration is read from the source --
    which is also what a reviewer would compare by eye, and does not go stale
    when the parser is rearranged.
    """
    import ast

    fuente = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")
    declaradas: set[str] = set()
    for nodo in ast.walk(ast.parse(fuente)):
        if not isinstance(nodo, ast.Call):
            continue
        if not (nodo.args and isinstance(nodo.args[0], ast.Constant)
                and nodo.args[0].value == "--only"):
            continue
        for palabra in nodo.keywords:
            if palabra.arg == "choices":
                declaradas = set(ast.literal_eval(palabra.value))
    assert declaradas, "the --only argument no longer declares its choices"

    almacenadas = {fila[0] for fila in
                   lexico.execute("SELECT DISTINCT source FROM document")}
    for eleccion in declaradas:
        assert eleccion.upper() in almacenadas, (
            f"the CLI offers {eleccion!r}, which no document is stored as")
