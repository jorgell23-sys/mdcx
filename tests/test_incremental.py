"""Checks that packaging costs what was added rather than what the corpus holds.

Indexing meaning dominates the cost of packaging: on one measured book, 504
seconds of encoding against 4 of compression and 0.2 of encryption. Encoding the
whole corpus on every publication makes adding one document cost a reindex of
every previous one, so the price of publishing grows with what has accumulated
instead of with what is new.

A passage whose text has not changed has the same vector, so it is recognised by
the digest of its text and read from the previous package, which already holds it
and is already encrypted with the user's key.

The tests that need the model are skipped without the multilingual extra, which
is a supported configuration.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, semantic  # noqa: E402

needs_model = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")


def write_corpus(folder: Path, count: int, start: int = 1) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + count):
        (folder / f"{i:03d}_doc.md").write_text(
            f"---\nsource_format: pdf\n---\n\n"
            f"Document {i}. Photosynthesis lets plants turn sunlight into sugar, "
            f"and this passage exists so that document {i} has content of its own.\n",
            encoding="utf-8")


def test_the_digest_identifies_a_passage():
    """The same text has the same identity; any edit has a different one."""
    assert archive.passage_digest("a passage") == archive.passage_digest("a passage")
    assert archive.passage_digest("a passage") != archive.passage_digest("a passage.")


@needs_model
def test_unchanged_passages_are_not_encoded_again():
    """Adding documents encodes the new ones only."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recibidos = root / "corpus" / "Received"
        write_corpus(recibidos, 6)
        first = archive.pack(root / "corpus", root / "v1.mdcx", "k", semantic=True)
        assert first["passages_reused"] == 0
        assert first["passages_encoded"] == first["passages"]

        write_corpus(recibidos, 2, start=7)
        second = archive.pack(root / "corpus", root / "v2.mdcx", "k", semantic=True,
                              reuse_from=root / "v1.mdcx")
        assert second["passages"] > first["passages"]
        assert second["passages_reused"] == first["passages"]
        assert second["passages_encoded"] == second["passages"] - first["passages"]


@needs_model
def test_reuse_produces_equivalent_retrieval():
    """A reused vector ranks as the vector it replaces.

    Half precision means a reused vector and a freshly encoded one can differ in
    the last bit. What must not differ is the ranking they produce.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recibidos = root / "corpus" / "Received"
        write_corpus(recibidos, 8)
        archive.pack(root / "corpus", root / "v1.mdcx", "k", semantic=True)
        write_corpus(recibidos, 2, start=9)

        archive.pack(root / "corpus", root / "full.mdcx", "k", semantic=True)
        archive.pack(root / "corpus", root / "reused.mdcx", "k", semantic=True,
                     reuse_from=root / "v1.mdcx")

        completo, _ = archive.open_package(root / "full.mdcx", "k")
        reutilizado, _ = archive.open_package(root / "reused.mdcx", "k")
        for consulta in ("photosynthesis sugar", "document 3", "sunlight plants"):
            a = [r["document"] for r in archive.query(completo, consulta, limit=5)]
            b = [r["document"] for r in archive.query(reutilizado, consulta, limit=5)]
            assert a == b, f"ranking differs for {consulta!r}"


@needs_model
def test_an_edited_passage_is_encoded_again():
    """Reuse follows the text. A changed document does not keep its vector."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recibidos = root / "corpus" / "Received"
        write_corpus(recibidos, 4)
        first = archive.pack(root / "corpus", root / "v1.mdcx", "k", semantic=True)

        objetivo = recibidos / "002_doc.md"
        objetivo.write_text(objetivo.read_text(encoding="utf-8").replace(
            "Photosynthesis", "Respiration"), encoding="utf-8")

        second = archive.pack(root / "corpus", root / "v2.mdcx", "k", semantic=True,
                              reuse_from=root / "v1.mdcx")
        assert second["passages_encoded"] >= 1, "the edited passage must be encoded"
        assert second["passages_reused"] == first["passages"] - second["passages_encoded"]


@needs_model
def test_a_package_from_another_model_is_not_reused():
    """Vectors from two models occupy different spaces and cannot be mixed.

    Reusing them would rank over quantities that are not comparable, which is a
    failure that produces plausible results rather than an error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recibidos = root / "corpus" / "Received"
        write_corpus(recibidos, 4)
        archive.pack(root / "corpus", root / "v1.mdcx", "k", semantic=True)

        reutilizables = archive.reusable_vectors(root / "v1.mdcx", "k",
                                                 "another/model-entirely")
        assert reutilizables == {}


def test_reuse_of_a_package_without_vectors_is_empty():
    """A package built without meaning holds nothing to reuse, and says so."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_corpus(root / "corpus" / "Received", 3)
        archive.pack(root / "corpus", root / "plain.mdcx", "k")
        assert archive.reusable_vectors(root / "plain.mdcx", "k",
                                        semantic.model_name()) == {}
