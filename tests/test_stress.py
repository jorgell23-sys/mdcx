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

"""Stress tests: hostile inputs and edge cases.

The package is expected to run on machines other than the one it was built on,
against documents it has never seen. These tests supply the inputs a real
collection eventually contains: empty files, corrupted files, names in other
alphabets, wrong extensions, and packages that have been truncated or tampered
with.

The requirement is not that every input succeeds. It is that no input causes an
unhandled crash, silent data loss, or a result presented as valid when it is not.

    python -m pytest tests/ -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdcx import archive, search  # noqa: E402
from mdcx.convert import compact, verify  # noqa: E402

KEY = "stress-test-key"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _corpus(root: Path) -> Path:
    folder = root / "Received"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.md").write_text(
        "---\nsource_format: docx\n---\n\n"
        "For basic engineering, pipes of DN 50 or larger shall be modelled.\n",
        encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Documents with unusual names and contents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "name with accents aeiou nN.md",
    "file with  double  spaces.md",
    "dots.in.the.name.md",
    "UPPERCASE.md",
])
def test_unusual_file_names_load(workspace, name):
    folder = workspace / "Received"
    folder.mkdir(parents=True)
    (folder / name).write_text("Minimum pipe diameter is DN 50.\n", encoding="utf-8")
    docs = search.load_documents(workspace)
    assert len(docs) == 1
    assert docs[0]["pseudopath"].startswith("@/")


def test_non_latin_file_name(workspace):
    folder = workspace / "Received"
    folder.mkdir(parents=True)
    try:
        (folder / "文档.md").write_text("Pipe DN 50.\n", encoding="utf-8")
    except (OSError, UnicodeEncodeError):
        pytest.skip("file system does not accept this name")
    assert len(search.load_documents(workspace)) == 1


def test_control_characters_do_not_break_normalisation():
    assert isinstance(search._normalize("a\x00b\x07c\x1b[31md"), str)


def test_very_long_single_line(workspace):
    folder = workspace / "Received"
    folder.mkdir(parents=True)
    (folder / "long.md").write_text("word " * 100_000, encoding="utf-8")
    docs = search.load_documents(workspace)
    assert len(docs) == 1
    assert isinstance(search.rank_passages(docs, "word", limit=3), list)


def test_empty_document(workspace):
    folder = workspace / "Received"
    folder.mkdir(parents=True)
    (folder / "empty.md").write_text("", encoding="utf-8")
    assert isinstance(search.load_documents(workspace), list)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "",
    "   ",
    "?",
    "a",
    "the and of",
    "\x00",
    "x'; DROP TABLE document; --",
    "*" * 300,
    "ñ",
    "emoji \U0001F600 query",
])
def test_hostile_queries_return_a_list(workspace, query):
    folder = workspace / "Received"
    folder.mkdir(parents=True)
    (folder / "doc.md").write_text("Minimum pipe diameter DN 50.", encoding="utf-8")
    docs = search.load_documents(workspace)
    assert isinstance(search.search(docs, query, limit=3), list)


def test_empty_corpus(workspace):
    assert search.load_documents(workspace) == []
    assert search.search([], "anything", limit=5) == []


# ---------------------------------------------------------------------------
# Package integrity
# ---------------------------------------------------------------------------

def test_pack_open_roundtrip(workspace):
    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY, issuer="test")
    connection, header = archive.open_package(workspace / "o.mdcx", KEY)
    assert header["documents"] == 1
    assert header["_intact"] is True
    assert isinstance(archive.query(connection, "minimum diameter", limit=2), list)


def test_wrong_key_is_rejected(workspace):
    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY)
    with pytest.raises(ValueError):
        archive.open_package(workspace / "o.mdcx", "wrong-key")


def test_tampered_body_is_detected(workspace):
    target = workspace / "o.mdcx"
    archive.pack(_corpus(workspace / "c"), target, KEY)
    raw = bytearray(target.read_bytes())
    raw[-100] ^= 0xFF
    target.write_bytes(bytes(raw))
    assert archive.read_header(target)["_intact"] is False
    with pytest.raises(ValueError):
        archive.open_package(target, KEY)


def test_truncated_package_is_rejected(workspace):
    target = workspace / "o.mdcx"
    archive.pack(_corpus(workspace / "c"), target, KEY)
    raw = target.read_bytes()
    target.write_bytes(raw[: len(raw) // 2])
    with pytest.raises((ValueError, EOFError, OSError)):
        archive.open_package(target, KEY)


def test_random_bytes_are_not_a_package(workspace):
    target = workspace / "random.mdcx"
    target.write_bytes(os.urandom(4096))
    with pytest.raises(ValueError):
        archive.read_header(target)


def test_empty_file_is_not_a_package(workspace):
    target = workspace / "empty.mdcx"
    target.write_bytes(b"")
    with pytest.raises((ValueError, OSError)):
        archive.read_header(target)


def test_export_restores_documents(workspace):
    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY)
    result = archive.export(workspace / "o.mdcx", KEY, workspace / "restored")
    assert result["documents"] == 1
    assert list((workspace / "restored").rglob("*.md"))


def test_header_is_readable_without_the_key(workspace):
    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY, issuer="acme")
    header = archive.read_header(workspace / "o.mdcx")
    assert header["issuer"] == "acme"
    assert header["_intact"] is True


# ---------------------------------------------------------------------------
# Compaction must never lose content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "---\nsource_format: pdf\n---\n\nPlain paragraph.\n",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "| a |  |  |\n|---|---|---|\n| 1 |  |  |\n",
    "<!-- image -->\n\nText after a marker.\n",
    "Unicode: aeiou 123 !@#$\n",
    "|broken table\n|---\n",
    "\n\n\n\n",
])
def test_compaction_never_loses_words(text):
    compacted, _ = compact.compact(text)
    before = verify.tokenize(verify.strip_markdown_noise(text))
    after = verify.tokenize(verify.strip_markdown_noise(compacted))
    missing = before - after
    assert not missing, f"compaction lost: {list(missing)[:5]}"


def test_compaction_is_idempotent():
    text = "| a | b |\n|-------|-------|\n| 1 | 2 |\n\n\n\nEnd.\n"
    once, _ = compact.compact(text)
    twice, _ = compact.compact(once)
    assert once == twice


def test_compaction_keeps_provenance_field():
    text = "---\nsource: x.pdf\nsource_format: pdf\nextra: dropped\n---\n\nBody.\n"
    compacted, _ = compact.compact(text)
    assert "source_format" in compacted


# ---------------------------------------------------------------------------
# Issuer signing
# ---------------------------------------------------------------------------

def test_signed_package_verifies_with_its_public_key(workspace):
    private, public = archive.generate_signing_key()
    archive.pack(_corpus(workspace / "c"), workspace / "s.mdcx", KEY,
                 issuer="acme", signing_key=private)
    assert archive.verify_signature(workspace / "s.mdcx", public) is True


def test_signature_fails_with_a_different_key(workspace):
    private, _ = archive.generate_signing_key()
    _, other = archive.generate_signing_key()
    archive.pack(_corpus(workspace / "c"), workspace / "s.mdcx", KEY,
                 signing_key=private)
    assert archive.verify_signature(workspace / "s.mdcx", other) is False


def test_unsigned_package_does_not_verify(workspace):
    _, public = archive.generate_signing_key()
    archive.pack(_corpus(workspace / "c"), workspace / "u.mdcx", KEY)
    assert archive.verify_signature(workspace / "u.mdcx", public) is False


def test_signature_rejects_a_tampered_body(workspace):
    """A signature covering only the recorded digest would accept a replaced body."""
    private, public = archive.generate_signing_key()
    target = workspace / "s.mdcx"
    archive.pack(_corpus(workspace / "c"), target, KEY, signing_key=private)
    raw = bytearray(target.read_bytes())
    raw[-50] ^= 0xFF
    target.write_bytes(bytes(raw))
    assert archive.verify_signature(target, public) is False


def test_malformed_public_key_is_rejected(workspace):
    private, _ = archive.generate_signing_key()
    archive.pack(_corpus(workspace / "c"), workspace / "s.mdcx", KEY,
                 signing_key=private)
    assert archive.verify_signature(workspace / "s.mdcx", "not-a-key") is False


def test_generated_keys_are_distinct():
    a, _ = archive.generate_signing_key()
    b, _ = archive.generate_signing_key()
    assert a != b


def test_pack_refuses_an_empty_corpus(workspace):
    """An empty package written without complaint fails only when queried."""
    empty = workspace / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No Markdown documents"):
        archive.pack(empty, workspace / "o.mdcx", KEY)


def test_pack_refuses_a_missing_folder(workspace):
    with pytest.raises(ValueError, match="Not a folder"):
        archive.pack(workspace / "nope", workspace / "o.mdcx", KEY)


# ---------------------------------------------------------------------------
# Concurrency
#
# The MCP SDK dispatches synchronous handlers on a thread pool, so a cached
# connection is consumed from a different thread on each call. Reported against
# 1.0.2, where the server served one call per restart.
# ---------------------------------------------------------------------------

def test_connection_is_usable_from_other_threads(workspace):
    import threading

    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY)
    connection, _ = archive.open_package(workspace / "o.mdcx", KEY)

    results, errors = [], []

    def run():
        try:
            results.append(archive.query(connection, "minimum diameter", limit=2))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"query failed across threads: {errors[0]}"
    assert len(results) == 8


def test_concurrent_queries_return_consistent_results(workspace):
    """Allowing cross-thread use without a lock returns wrong rows without raising."""
    import threading

    archive.pack(_corpus(workspace / "c"), workspace / "o.mdcx", KEY)
    connection, _ = archive.open_package(workspace / "o.mdcx", KEY)

    expected = len(archive.query(connection, "minimum diameter", limit=5))
    counts, errors = [], []

    def run():
        for _ in range(10):
            try:
                counts.append(len(archive.query(connection, "minimum diameter", limit=5)))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert set(counts) == {expected}, f"divergent results: {sorted(set(counts))}"


def test_statistics_cache_is_not_keyed_by_object_id(workspace):
    """id() is recycled, which could alias a stale entry from a closed package."""
    archive.pack(_corpus(workspace / "a"), workspace / "a.mdcx", KEY)

    other = workspace / "b"
    (other / "Received").mkdir(parents=True)
    (other / "Received" / "x.md").write_text(
        "---\nsource_format: pdf\n---\n\nCompletely different wording here.\n",
        encoding="utf-8")
    archive.pack(other, workspace / "b.mdcx", KEY)

    first, _ = archive.open_package(workspace / "a.mdcx", KEY)
    first.close()
    del first

    second, _ = archive.open_package(workspace / "b.mdcx", KEY)
    assert isinstance(archive.query(second, "different wording", limit=2), list)


def test_mcp_tools_are_async():
    """Synchronous handlers are dispatched on a thread pool by the SDK."""
    import inspect

    pytest.importorskip("mcp")
    from mdcx import mcp_server

    source = inspect.getsource(mcp_server.create_server)
    for tool in ("def search(", "def info(", "def document("):
        assert f"async {tool}" in source, f"{tool} must be async"
