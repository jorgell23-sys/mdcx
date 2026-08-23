"""Checks that what the package declares can actually be started.

Two defects in 1.3.2 shared a shape. `mdcx --help` died with NameError because
main() reached for a module that only an inner function had imported, and the
MCP tool `info` published the signature of an internal helper because the
decorator had stayed on it. Both survived a full test run: one lives in the
console script, the other in the tool registry, and nothing crossed either
boundary. The suite called the API directly, where both were correct.

So these tests cross the two boundaries. Every console script declared in
pyproject.toml is executed as a process, and every tool is read back from the
server that registers it rather than from the source that defines it.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mdcx import archive, mcp_server  # noqa: E402

SCRIPTS = ("mdcx", "mdcx-convert", "mdcx-search")


def _declared_scripts() -> dict[str, str]:
    """The console scripts the package promises, read from pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 has no tomllib.
        pytest.skip("reading pyproject.toml needs tomllib (3.11+)")
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def test_the_declared_scripts_are_the_ones_checked_here():
    """A fourth entry point should arrive with a smoke test of its own."""
    assert set(_declared_scripts()) == set(SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_console_script_starts(script):
    """--help must reach the user, and the exit code must say that it did.

    This is the cheapest check that would have caught the 1.3.2 regression:
    `mdcx` raised NameError before parsing a single argument, so all six of its
    subcommands were unreachable — pack among them, the one that produces the
    package — while every test of the API stayed green.

    It runs as a separate process because that is how a console script runs,
    and because an import inside this one would hide a missing import there.
    """
    module, function = _declared_scripts()[script].split(":")
    done = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.argv = ['{script}', '--help']\n"
         f"from {module} import {function}\n"
         f"raise SystemExit({function}())"],
        capture_output=True, text=True, timeout=180,
        env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONIOENCODING="utf-8"))
    assert done.returncode == 0, f"{script} --help exited {done.returncode}: {done.stderr}"
    assert "usage" in done.stdout.lower()


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_console_script_reports_its_version(script):
    """SECURITY.md asks a reporter to quote the version, so it has to be gettable.

    It asked for `mdcx --version`, which parsed as a missing subcommand and
    exited 2. A policy that names a command is a promise that the command
    exists, and nothing here was checking that one did.
    """
    module, function = _declared_scripts()[script].split(":")
    done = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.argv = ['{script}', '--version']\n"
         f"from {module} import {function}\n"
         f"raise SystemExit({function}())"],
        capture_output=True, text=True, timeout=180,
        env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONIOENCODING="utf-8"))
    assert done.returncode == 0, f"{script} --version exited {done.returncode}: {done.stderr}"
    assert "mdcx" in done.stdout.lower()


@pytest.fixture
def one_package(tmp_path, monkeypatch):
    """A package with a single document, configured as the served corpus."""
    received = tmp_path / "a" / "Received"
    received.mkdir(parents=True)
    (received / "press.md").write_text(
        "---\nsource_format: pdf\n---\n\nGutenberg built the printing press.\n",
        encoding="utf-8")
    archive.pack(tmp_path / "a", tmp_path / "only.mdcx", "k")
    monkeypatch.setenv("MDCX_FILE", str(tmp_path / "only.mdcx"))
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    yield tmp_path
    mcp_server._STATE.clear()


def test_every_tool_publishes_the_signature_of_the_function_that_answers_it():
    """A decorator on a helper advertises the helper.

    That is what happened to `info`: the schema reaching the client was
    `_describeArguments`, requiring a `paquete` object — the server's own open
    connection and decrypted header. No client can build one, so the tool could
    only fail, and the published schema said so without running anything.
    """
    server = mcp_server.create_server()
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"search", "info", "document"}
    for tool in tools:
        schema = tool.input_schema
        assert schema.get("title") == f"{tool.name}Arguments", (
            f"{tool.name} publishes {schema.get('title')}: the decorator is on "
            f"another function")
        for name, prop in (schema.get("properties") or {}).items():
            assert prop.get("type") != "object", (
                f"{tool.name} takes {name}, an object no client can construct")


def test_info_answers_when_called_with_no_arguments(one_package):
    """The server tells the client to call info first, so it must be callable.

    The instructions sent to the client say to use info to learn what the
    corpus contains before querying it, and search points to its
    cross_language_search field as the only way to read an empty result. An
    agent that follows those instructions makes this exact call.
    """
    server = mcp_server.create_server()
    respuesta = asyncio.run(server.call_tool("info", {}))

    assert not respuesta.is_error, respuesta.content
    registro = json.loads(respuesta.content[0].text)
    assert registro["documents"] == 1
    assert registro["integrity"] == "intact"
    assert "cross_language_search" in registro
