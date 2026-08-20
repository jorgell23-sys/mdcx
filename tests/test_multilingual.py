"""Measures retrieval across languages, which word matching cannot do.

A query written in Spanish and a document written in German about the same
subject share no word, so a lexical index has nothing to match however well it
tokenises. These tests check the property that makes such a document reachable:
the package indexes meaning as well as words, and the two are merged.

They are skipped when the multilingual extra is absent, because without it the
package retrieves by word alone, which is a supported configuration rather than
a failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import semantic  # noqa: E402

DATA = Path(__file__).parent / "data"
SUBJECTS = ("math", "bio", "hist", "food")

pytestmark = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")


def load_corpus() -> dict:
    corpus: dict = {}
    for name in ("languages_latin.json", "languages_other.json"):
        corpus.update(json.loads((DATA / name).read_text(encoding="utf-8")))
    return corpus


CORPUS = load_corpus()


@pytest.fixture(scope="module")
def package():
    """One corpus in every language, indexed by word and by meaning."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "corpus" / "Received"
        folder.mkdir(parents=True)
        for language, entry in CORPUS.items():
            for subject in SUBJECTS:
                (folder / f"{language}__{subject}.md").write_text(
                    f"---\nsource_format: pdf\n---\n\n{entry['docs'][subject]}\n",
                    encoding="utf-8")
        target = root / "multilingual.mdcx"
        archive.pack(root / "corpus", target, "test-key", semantic=True)
        connection, header = archive.open_package(target, "test-key")
        yield connection, header


def languages_of(results: list[dict]) -> list[str]:
    return [r["document"].split("__")[0].split("/")[-1] for r in results]


def test_package_records_the_model(package):
    """The package names the model that produced its vectors.

    A query encoded by a different model lands elsewhere in the space, and
    comparing the two would return confident nonsense.
    """
    connection, _ = package
    recorded = connection.execute(
        "SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    assert recorded and recorded[0] == semantic.model_name()
    assert archive.has_vectors(connection)


def test_query_reaches_other_languages(package):
    """A query written in one language returns documents written in others."""
    connection, _ = package
    results = archive.query(connection, CORPUS["spanish"]["queries"]["hist"], limit=8)
    found = set(languages_of(results))
    assert "spanish" in found, "the language of the query must still be served"
    assert len(found - {"spanish"}) >= 3, (
        f"only reached {sorted(found)}; a query should cross into other languages")


def test_lexical_mode_stays_within_the_language(package):
    """The lexical engine is unchanged and still available on its own.

    This is the comparison that shows what meaning adds: the same query, the
    same package, one engine reaching across languages and the other not.
    """
    connection, _ = package
    consulta = CORPUS["spanish"]["queries"]["hist"]
    lexical = set(languages_of(archive.query(connection, consulta, limit=8, mode="lexical")))
    fused = set(languages_of(archive.query(connection, consulta, limit=8, mode="auto")))
    assert len(fused) > len(lexical)


def test_fusion_keeps_the_own_language_first(package):
    """Merging must not cost the precision the lexical engine already had.

    Used alone the dense engine ranks a related document above the exact one
    often enough to matter. The merge exists so that neither property is traded
    for the other.
    """
    connection, _ = package
    first = 0
    for language in ("english", "spanish", "german", "russian", "chinese"):
        for subject in SUBJECTS:
            results = archive.query(connection, CORPUS[language]["queries"][subject],
                                    limit=3)
            if results and results[0]["document"].endswith(f"{language}__{subject}"):
                first += 1
    assert first >= 18, f"the expected document came first in only {first} of 20"


@pytest.mark.parametrize("language", ["german", "russian", "chinese", "arabic", "hindi"])
def test_every_script_is_reachable_from_spanish(package, language):
    """A Spanish query reaches the document of each writing system.

    Retrieval across languages must not be a property of the Latin alphabet.
    """
    connection, _ = package
    for subject in SUBJECTS:
        results = archive.query(connection, CORPUS["spanish"]["queries"][subject],
                                limit=34)
        alcanzados = {r["document"] for r in results}
        if f"{language}__{subject}" in alcanzados:
            return
    pytest.fail(f"no Spanish query reached any {language} document")


def test_a_package_without_vectors_still_answers():
    """A package built without meaning is queried by word, without failing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "corpus" / "Received"
        folder.mkdir(parents=True)
        (folder / "01_english.md").write_text(
            "---\nsource_format: pdf\n---\n\nGutenberg built the printing press.\n",
            encoding="utf-8")
        target = root / "plain.mdcx"
        archive.pack(root / "corpus", target, "k")
        connection, _ = archive.open_package(target, "k")
        assert not archive.has_vectors(connection)
        assert archive.query(connection, "printing press", limit=3)
        assert archive.query(connection, "printing press", limit=3, mode="auto")
        assert archive.query(connection, "printing press", limit=3, mode="semantic") == []
