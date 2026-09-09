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

"""What a server pays before it can answer anything.

Opening a package decompresses its body, which the code assumed was a fraction
of a second. On a 254 MB package a consumer measured 14.85 s, of which 94% is
the decompressor, and the server opened all 65 of theirs before creating the
server object: sixteen minutes, against a client that gives up at thirty
seconds. A corpus that grew past a certain size stopped being servable, and the
only symptom was a connection timeout that named nothing.

Each database is also held in memory, so opening everything asks for the whole
corpus decompressed at once -- 51 GB for that collection -- whether or not a
query ever touches it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, mcp_server  # noqa: E402


def _package(tmp_path, name, documents):
    folder = tmp_path / name
    folder.mkdir()
    for title, text in documents.items():
        (folder / f"{title}.md").write_text(f"# {title}\n\n{text}\n",
                                            encoding="utf-8")
    target = tmp_path / f"{name}.mdcx"
    archive.pack(folder, target, "k")
    return target


def _configure(monkeypatch, packages):
    monkeypatch.setattr(mcp_server, "_STATE", {})
    monkeypatch.setenv("MDCX_FILE", os.pathsep.join(str(p) for p in packages))
    monkeypatch.setenv("MDCX_KEY", "k")


# --- Nothing is opened to start ------------------------------------------------


def test_configuring_does_not_open_anything(tmp_path, monkeypatch):
    """The whole of the fault: the server opened the corpus before it existed
    as a server, so a large collection never reached `run()`."""
    first = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    second = _package(tmp_path, "two", {"b": "Beta text about presses."})
    _configure(monkeypatch, [first, second])

    packages = mcp_server._open_packages()

    assert len(packages) == 2
    assert not any(p.is_open() for p in packages), (
        "a package was opened merely by being configured")


def test_a_package_opens_when_it_is_read(tmp_path, monkeypatch):
    """Deferred, not skipped."""
    only = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    _configure(monkeypatch, [only])

    package = mcp_server._open_packages()[0]
    assert not package.is_open()

    assert package["header"]["documents"] == 1
    assert package.is_open()


def test_reading_a_package_twice_opens_it_once(tmp_path, monkeypatch):
    """Deferring must not turn into repeating: decompression is the expensive
    part and it is paid once per package, as before."""
    only = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    _configure(monkeypatch, [only])

    package = mcp_server._open_packages()[0]

    assert package["connection"] is package["connection"]


def test_a_missing_path_is_still_refused_at_startup(tmp_path, monkeypatch):
    """What a client cannot act on once connected is still settled up front:
    a path that is not there is a configuration error, not a query error."""
    _configure(monkeypatch, [tmp_path / "absent.mdcx"])

    try:
        mcp_server._open_packages()
    except RuntimeError as refused:
        assert "not found" in str(refused)
    else:
        raise AssertionError("a package that does not exist was accepted")


def test_a_key_that_does_not_open_it_is_found_when_queried(tmp_path, monkeypatch):
    """The other side of the trade, stated: a package that cannot be decrypted
    is discovered later than before -- and at the moment a client can report
    it, rather than as a connection that never completes."""
    only = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    monkeypatch.setattr(mcp_server, "_STATE", {})
    monkeypatch.setenv("MDCX_FILE", str(only))
    monkeypatch.setenv("MDCX_KEY", "the wrong key")

    package = mcp_server._open_packages()[0]

    try:
        package["header"]
    except Exception:
        pass
    else:
        raise AssertionError("the wrong key opened the package")


# --- What an open package costs ------------------------------------------------


def test_an_unopened_package_reports_no_memory(tmp_path, monkeypatch):
    """`None` rather than zero: it holds nothing because it was never read,
    which is not the same as holding nothing."""
    only = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    _configure(monkeypatch, [only])

    package = mcp_server._open_packages()[0]

    assert mcp_server._resident_mib(package) is None


def test_an_open_package_says_what_it_holds(tmp_path, monkeypatch):
    """How large a corpus one server can hold is bounded by memory, and the
    bound was nowhere: it was met rather than seen coming."""
    only = _package(tmp_path, "one", {"a": "Alpha text about plants."})
    _configure(monkeypatch, [only])

    package = mcp_server._open_packages()[0]
    package["connection"]

    assert mcp_server._resident_mib(package) is not None
    assert mcp_server._resident_mib(package) >= 0


# --- What compressed the body --------------------------------------------------


def test_a_package_says_what_compressed_it(tmp_path):
    """The header has always carried the name; until now it could only say one
    thing, and a reader had no reason to look at it."""
    folder = tmp_path / "md"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\nProse to compress.\n", encoding="utf-8")

    written = archive.pack(folder, tmp_path / "c.mdcx", "k")

    assert written["compression"] == "lzma"
    _connection, header = archive.open_package(tmp_path / "c.mdcx", "k")
    assert header["compression"] == "lzma"


def test_a_package_written_with_zstd_reads_back(tmp_path):
    """Measured by a consumer on 784 MB of their own database: LZMA wrote
    254 MB and took 14.01 s to read back, zstd wrote 233 MB and took 1.14 s.
    Opening is what a server pays before it can answer anything."""
    zstandard = pytest.importorskip("zstandard")
    del zstandard

    folder = tmp_path / "md"
    folder.mkdir()
    for i in range(20):
        (folder / f"d{i}.md").write_text(
            f"# Doc {i}\n\nProse about topic {i}, repeated for compression.\n",
            encoding="utf-8")

    written = archive.pack(folder, tmp_path / "c.mdcx", "k", compression="zstd")
    connection, header = archive.open_package(tmp_path / "c.mdcx", "k")

    assert written["compression"] == "zstd"
    assert header["compression"] == "zstd"
    assert archive.query(connection, "prose topic", limit=3)


def test_a_package_with_no_name_is_read_as_lzma(tmp_path):
    """Written before the header could say anything else. Defaulting to what it
    actually was is the difference between opening it and refusing it."""
    assert archive._decompress(
        __import__("lzma").compress(b"a database would go here"), None) == (
            b"a database would go here")


def test_a_compressor_this_version_cannot_read_says_so(tmp_path):
    """Rather than failing inside a decompressor with a message about bytes."""
    try:
        archive._decompress(b"anything", "brotli")
    except ValueError as refused:
        assert "brotli" in str(refused)
    else:
        raise AssertionError("an unknown compressor was attempted anyway")


def test_a_compressor_that_does_not_exist_is_refused_before_indexing(tmp_path):
    """Indexing a corpus and then failing to seal it wastes the expensive
    half, so the name is checked against the list first."""
    folder = tmp_path / "md"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\nProse.\n", encoding="utf-8")

    try:
        archive.pack(folder, tmp_path / "c.mdcx", "k", compression="brotli")
    except ValueError as refused:
        assert "lzma" in str(refused) and "zstd" in str(refused)
    else:
        raise AssertionError("an unknown compressor was accepted")

