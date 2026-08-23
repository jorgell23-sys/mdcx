"""Checks that what the server says about itself matches what it returns.

The description of a tool is not documentation. It is the instruction a model
reads before deciding how to ask, and the only account of the reply it will
ever see -- so a description that has fallen behind the code does not merely
mislead a reader, it misleads the caller at the moment of calling.

That is what happened. Behaviour changed twice and the descriptions did not
follow: `search` still told agents to write their query in the language of the
documents after a query in another language had begun to be answered by
meaning, and still described a passage as carrying a score after the score had
been replaced by a rank. Nothing failed, because nothing here was checked.

These tests are about correspondence, not wording. They ask that every field a
reply actually contains be accounted for somewhere in the text the caller is
given, and that the text not promise fields that no longer arrive.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, mcp_server, semantic  # noqa: E402

pytestmark = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")

# Fields a caller has to understand to read a reply, and would misuse if the
# description stayed silent about them. Some belong to the reply and some to
# each passage in it, which is itself part of what the description has to say.
EN_LA_RESPUESTA = ("similarity",)
EN_CADA_PASAJE = ("rank",)
EN_INFO = ("documents", "integrity")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A package with enough in it to answer and to fail to answer."""
    raiz = tmp_path_factory.mktemp("dice")
    recibidos = raiz / "src" / "Received"
    recibidos.mkdir(parents=True)
    (recibidos / "chemistry.md").write_text(
        "---\nsource_format: pdf\n---\n\n# Solutions\n\n"
        "A saturated solution holds as much solute as it can at that "
        "temperature. Adding more leaves it undissolved at the bottom.\n\n"
        "# Acids\n\nAn acid donates a proton and a base accepts one. "
        "Titration finds the concentration of one using the other.\n",
        encoding="utf-8")
    paquete = raiz / "corpus.mdcx"
    archive.pack(raiz / "src", paquete, "k", semantic=True)
    os.environ["MDCX_FILE"] = str(paquete)
    os.environ["MDCX_KEY"] = "k"
    mcp_server._STATE.clear()
    yield mcp_server.create_server()
    mcp_server._STATE.clear()


def _described(server, name: str) -> str:
    """Everything the caller is told about one tool, as one body of text."""
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        if tool.name == name:
            return f"{tool.title or ''}\n{tool.description or ''}"
    raise AssertionError(f"la herramienta {name} no se publica")


def _replied(server, name: str, arguments: dict) -> dict:
    salida = asyncio.run(server.call_tool(name, arguments))
    assert not salida.is_error, salida.content
    return json.loads(salida.content[0].text)


def test_every_field_of_a_reply_is_accounted_for(served):
    """A caller cannot use what it was never told arrives."""
    respuesta = _replied(served, "search", {"query": "what is a saturated solution"})
    texto = _described(served, "search").lower()
    assert respuesta["passages"], "hace falta al menos un pasaje para comprobarlo"

    for campo in EN_LA_RESPUESTA:
        assert campo in respuesta, f"la respuesta ya no trae {campo}"
        assert campo in texto, (
            f"la respuesta trae '{campo}' y la descripcion no lo menciona")

    for campo in EN_CADA_PASAJE:
        assert campo in respuesta["passages"][0], f"un pasaje ya no trae {campo}"
        assert campo in texto, (
            f"un pasaje trae '{campo}' y la descripcion no lo menciona")


def test_every_argument_a_tool_accepts_is_explained(served):
    """A parameter the caller can see but cannot understand is worse than none.

    The schema publishes the name and the type; neither says what values are
    accepted. `direction` was declared, took exactly two words, silently ignored
    everything else, and appeared in no description -- so a caller could only
    guess, and a wrong guess widened the search without saying so.

    The name alone in the description is the bar here. What the values are is
    prose, and prose is not something a test can judge; that a caller is told
    the parameter exists at all is.
    """
    tools = asyncio.run(served.list_tools())
    assert tools, "the server publishes no tools"
    for tool in tools:
        texto = f"{tool.title or ''} {tool.description or ''}".lower()
        declarados = (tool.input_schema or {}).get("properties", {})
        for nombre in declarados:
            assert nombre.lower() in texto, (
                f"{tool.name} accepts {nombre!r} and never mentions it")


def test_the_description_does_not_promise_a_field_that_stopped_arriving(served):
    """The score was replaced by a rank, and saying otherwise invites sorting by it."""
    respuesta = _replied(served, "search", {"query": "what is an acid"})
    assert "score" not in respuesta["passages"][0]
    texto = _described(served, "search").lower()
    assert "each with its score" not in texto
    assert "rank" in texto, "el orden es lo unico que la fusion establece"


def test_the_warning_is_described_where_it_can_appear(served):
    """It changes what the reply means, so it cannot arrive unannounced."""
    respuesta = _replied(served, "search", {"query": "premier league offside rule"})
    texto = _described(served, "search").lower()
    if "warning" in respuesta:
        assert "warning" in texto, (
            "la respuesta puede traer un aviso que la descripcion no anuncia")


def test_info_describes_what_it_answers(served):
    """It is the tool the server tells the caller to use first."""
    respuesta = _replied(served, "info", {})
    texto = _described(served, "info").lower()
    for campo in EN_INFO:
        assert campo in respuesta, f"info ya no informa {campo}"
    assert "document" in texto and "conver" in texto


def test_the_server_instructions_name_the_tools_they_send_you_to(served):
    """Instructions that name a tool that is not published send the caller nowhere."""
    publicadas = {t.name for t in asyncio.run(served.list_tools())}
    fuente = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "mcp_server.py").read_text(encoding="utf-8")
    inicio = fuente.index("instructions=(")
    instrucciones = fuente[inicio:inicio + 600].lower()
    for nombre in ("search", "info"):
        assert nombre in instrucciones
        assert nombre in publicadas, f"las instrucciones mandan a {nombre}, que no existe"
