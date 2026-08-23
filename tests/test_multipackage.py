"""Checks that the server can query several packages as one corpus.

A corpus that grows daily cannot live in a single file. Rebuilding it costs what
has accumulated rather than what is new, and opening it decrypts the whole of it
into memory, so the size of the file imposes a ceiling before anything else does.

Querying several packages makes each batch immutable: it is indexed once and
never touched again. Publishing a day costs a day.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, mcp_server  # noqa: E402


def build(folder: Path, target: Path, key: str, documents: dict[str, str]) -> Path:
    received = folder / "Received"
    received.mkdir(parents=True, exist_ok=True)
    for name, text in documents.items():
        (received / f"{name}.md").write_text(
            f"---\nsource_format: pdf\n---\n\n{text}\n", encoding="utf-8")
    archive.pack(folder, target, key)
    return target


@pytest.fixture
def two_packages(tmp_path, monkeypatch):
    """Two packages holding different documents, configured as one corpus."""
    build(tmp_path / "a", tmp_path / "first.mdcx", "k",
          {"gutenberg": "Gutenberg built the printing press with movable type."})
    build(tmp_path / "b", tmp_path / "second.mdcx", "k",
          {"photosynthesis": "Photosynthesis lets plants turn sunlight into sugar."})
    monkeypatch.setenv("MDCX_FILE",
                       f"{tmp_path / 'first.mdcx'}{os.pathsep}{tmp_path / 'second.mdcx'}")
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    yield tmp_path
    mcp_server._STATE.clear()


def test_a_setting_names_several_packages():
    """Paths are separated the way the platform separates lists of paths.

    A comma is accepted too, since it is what people write, and a Windows drive
    letter survives either way.

    The drive letter is only checked where a drive letter exists. On a system
    whose path separator is the colon, "C:\\one.mdcx" is two paths and reading
    it as one would be the error -- this test asserted otherwise and passed
    only because it had never run anywhere but Windows.
    """
    if os.pathsep == ";":
        parts = mcp_server._split_setting(f"C:\\one.mdcx{os.pathsep}D:\\two.mdcx")
        assert parts == ["C:\\one.mdcx", "D:\\two.mdcx"]
    else:
        parts = mcp_server._split_setting(f"/data/one.mdcx{os.pathsep}/data/two.mdcx")
        assert parts == ["/data/one.mdcx", "/data/two.mdcx"]
    assert mcp_server._split_setting("a.mdcx, b.mdcx") == ["a.mdcx", "b.mdcx"]
    assert mcp_server._split_setting("only.mdcx") == ["only.mdcx"]


def test_both_packages_are_opened(two_packages):
    packages = mcp_server._open_packages()
    assert len(packages) == 2
    assert {p["name"] for p in packages} == {"first.mdcx", "second.mdcx"}


def test_a_query_reaches_either_package(two_packages):
    """A document is found whichever package holds it, and names its own."""
    for query, expected, package in (
            ("printing press", "gutenberg", "first.mdcx"),
            ("photosynthesis sugar", "photosynthesis", "second.mdcx")):
        results = mcp_server.search_packages(query, limit=3)
        assert results, f"nothing found for {query!r}"
        assert any(expected in r["document"] for r in results)
        assert all(r.get("package") for r in results), (
            "a result must say which package it came from")
        assert any(r.get("package") == package for r in results)


def test_a_single_package_omits_the_package_field(tmp_path, monkeypatch):
    """With one package there is nothing to disambiguate, and nothing is added."""
    build(tmp_path / "a", tmp_path / "only.mdcx", "k",
          {"gutenberg": "Gutenberg built the printing press."})
    monkeypatch.setenv("MDCX_FILE", str(tmp_path / "only.mdcx"))
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    try:
        results = mcp_server.search_packages("printing press", limit=3)
        assert results
        assert "package" not in results[0]
    finally:
        mcp_server._STATE.clear()


def test_one_key_serves_every_package(tmp_path, monkeypatch):
    """A single key is the ordinary case and is applied to all of them."""
    build(tmp_path / "a", tmp_path / "one.mdcx", "shared", {"a": "First document."})
    build(tmp_path / "b", tmp_path / "two.mdcx", "shared", {"b": "Second document."})
    monkeypatch.setenv("MDCX_FILE",
                       f"{tmp_path / 'one.mdcx'}{os.pathsep}{tmp_path / 'two.mdcx'}")
    monkeypatch.setenv("MDCX_KEY", "shared")
    mcp_server._STATE.clear()
    try:
        assert len(mcp_server._open_packages()) == 2
    finally:
        mcp_server._STATE.clear()


def test_a_key_per_package_is_matched_in_order(tmp_path, monkeypatch):
    build(tmp_path / "a", tmp_path / "one.mdcx", "first-key", {"a": "First document."})
    build(tmp_path / "b", tmp_path / "two.mdcx", "second-key", {"b": "Second document."})
    monkeypatch.setenv("MDCX_FILE",
                       f"{tmp_path / 'one.mdcx'}{os.pathsep}{tmp_path / 'two.mdcx'}")
    monkeypatch.setenv("MDCX_KEY", f"first-key{os.pathsep}second-key")
    mcp_server._STATE.clear()
    try:
        assert len(mcp_server._open_packages()) == 2
    finally:
        mcp_server._STATE.clear()


def test_a_mismatched_number_of_keys_is_reported(tmp_path, monkeypatch):
    """Two keys for three packages is a configuration error, not a guess."""
    build(tmp_path / "a", tmp_path / "one.mdcx", "k", {"a": "First."})
    build(tmp_path / "b", tmp_path / "two.mdcx", "k", {"b": "Second."})
    build(tmp_path / "c", tmp_path / "three.mdcx", "k", {"c": "Third."})
    monkeypatch.setenv("MDCX_FILE", os.pathsep.join(
        str(tmp_path / n) for n in ("one.mdcx", "two.mdcx", "three.mdcx")))
    monkeypatch.setenv("MDCX_KEY", f"k{os.pathsep}k")
    mcp_server._STATE.clear()
    try:
        with pytest.raises(RuntimeError, match="one key"):
            mcp_server._open_packages()
    finally:
        mcp_server._STATE.clear()


def test_a_missing_package_is_reported_by_name(tmp_path, monkeypatch):
    build(tmp_path / "a", tmp_path / "one.mdcx", "k", {"a": "First."})
    monkeypatch.setenv("MDCX_FILE",
                       f"{tmp_path / 'one.mdcx'}{os.pathsep}{tmp_path / 'absent.mdcx'}")
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    try:
        with pytest.raises(RuntimeError, match="absent.mdcx"):
            mcp_server._open_packages()
    finally:
        mcp_server._STATE.clear()


def test_a_single_package_still_works(tmp_path, monkeypatch):
    """The ordinary configuration is unchanged."""
    build(tmp_path / "a", tmp_path / "only.mdcx", "k",
          {"gutenberg": "Gutenberg built the printing press."})
    monkeypatch.setenv("MDCX_FILE", str(tmp_path / "only.mdcx"))
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    try:
        packages = mcp_server._open_packages()
        assert len(packages) == 1
        assert archive.query(packages[0]["connection"], "printing press", limit=3)
    finally:
        mcp_server._STATE.clear()
