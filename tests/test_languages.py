"""Measures retrieval across writing systems on a single mixed-language corpus.

The corpus holds the same four subjects — algebra, botany, printing and baking —
written in every language under test, so that each query competes against three
same-language documents on related subjects and against every other language at
once. A query is a few content words, not a copy of the sentence.

Two properties are measured.

Coverage: a query written in a language finds the document written in that
language. This is what makes a corpus usable in the language its documents were
written in.

Inclusion: a term that several languages share returns the documents of all of
them. Retrieval must not narrow to one language, whether the language of the
query or the predominant language of the corpus. A search returns what matches,
in whatever language it was written.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402

DATA = Path(__file__).parent / "data"
SUBJECTS = ("math", "bio", "hist", "food")


def load_corpus() -> dict:
    corpus: dict = {}
    for name in ("languages_latin.json", "languages_other.json"):
        corpus.update(json.loads((DATA / name).read_text(encoding="utf-8")))
    return corpus


CORPUS = load_corpus()

# Terms that several languages write identically. Each is a proper name or a
# loanword, which is the realistic case of a query that should reach beyond one
# language: the value is the least number of distinct languages that must appear
# in the results.
SHARED_TERMS = {
    "Gutenberg": 15,
    "Гутенберг": 4,
}


@pytest.fixture(scope="module")
def package():
    """Pack every language into one corpus and open it once for all tests."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "corpus" / "Received"
        folder.mkdir(parents=True)
        for language, entry in CORPUS.items():
            for subject in SUBJECTS:
                (folder / f"{language}__{subject}.md").write_text(
                    f"---\nsource_format: pdf\n---\n\n{entry['docs'][subject]}\n",
                    encoding="utf-8")

        target = root / "languages.mdcx"
        archive.pack(root / "corpus", target, "test-key")
        connection, header = archive.open_package(target, "test-key")
        yield connection, header


def rank_of(results: list[dict], language: str, subject: str) -> int:
    """Position of the expected document in the results, or 0 if absent."""
    wanted = f"{language}__{subject}"
    for position, result in enumerate(results, start=1):
        if wanted in result["document"]:
            return position
    return 0


def measure(connection) -> dict[str, dict]:
    """Rank of the expected document for every query, grouped by language."""
    report: dict[str, dict] = {}
    for language, entry in CORPUS.items():
        ranks = {}
        for subject in SUBJECTS:
            results = archive.query(connection, entry["queries"][subject], limit=5)
            ranks[subject] = rank_of(results, language, subject)
        report[language] = {
            "script": entry["script"],
            "code": entry["code"],
            "ranks": ranks,
            "first": sum(1 for r in ranks.values() if r == 1),
            "top3": sum(1 for r in ranks.values() if 1 <= r <= 3),
            "found": sum(1 for r in ranks.values() if r >= 1),
        }
    return report


def test_corpus_is_complete(package):
    """Every document reaches the index."""
    _, header = package
    assert header["documents"] == len(CORPUS) * len(SUBJECTS)


def test_every_language_finds_its_documents(package):
    """A query in a language retrieves the documents written in that language."""
    connection, _ = package
    report = measure(connection)
    failing = {name: data["ranks"] for name, data in report.items()
               if data["found"] < len(SUBJECTS)}
    assert not failing, f"queries that retrieved nothing: {failing}"


def test_ranking_is_precise(package):
    """The expected document is not merely present but ranked at the top."""
    connection, _ = package
    report = measure(connection)
    total = len(CORPUS) * len(SUBJECTS)
    first = sum(data["first"] for data in report.values())
    top3 = sum(data["top3"] for data in report.values())
    assert top3 == total, f"outside the top three: {total - top3} of {total}"
    assert first >= total * 0.9, f"first place in {first} of {total}"


@pytest.mark.parametrize("term,least_languages", sorted(SHARED_TERMS.items()))
def test_shared_terms_reach_every_language(package, term, least_languages):
    """A term several languages share returns the documents of all of them.

    Retrieval must not narrow to the language of the query or to the
    predominant language of the corpus.
    """
    connection, _ = package
    results = archive.query(connection, term, limit=60)
    languages = {r["document"].split("__")[0].split("/")[-1] for r in results}
    expected = {name for name, entry in CORPUS.items()
                if any(term in text for text in entry["docs"].values())}
    missing = expected - languages
    assert len(languages & expected) >= least_languages, (
        f"{term} reached {len(languages & expected)} languages, missing {sorted(missing)}")


def test_language_of_corpus_does_not_filter_results(package):
    """Documents in other languages stay reachable in a mixed corpus.

    The corpus records a predominant language. That record describes the corpus;
    it must never restrict what a query can return.
    """
    connection, header = package
    predominant = header.get("language")
    results = archive.query(connection, "Gutenberg", limit=60)
    languages = {r["document"].split("__")[0].split("/")[-1] for r in results}
    others = {name for name in languages
              if CORPUS.get(name, {}).get("code") != predominant}
    assert len(others) >= 10, (
        f"only {len(others)} languages other than {predominant} were returned")


if __name__ == "__main__":
    # The report prints every script, which a legacy console encoding cannot
    # represent. Writing UTF-8 directly keeps the output readable everywhere.
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "corpus" / "Received"
        folder.mkdir(parents=True)
        for language, entry in CORPUS.items():
            for subject in SUBJECTS:
                (folder / f"{language}__{subject}.md").write_text(
                    f"---\nsource_format: pdf\n---\n\n{entry['docs'][subject]}\n",
                    encoding="utf-8")
        target = root / "languages.mdcx"
        archive.pack(root / "corpus", target, "test-key")
        connection, header = archive.open_package(target, "test-key")

        report = measure(connection)
        width = max(len(name) for name in report)
        print(f"  Corpus: {header['documents']} documents, "
              f"{len(CORPUS)} languages, {len(set(e['script'] for e in CORPUS.values()))} scripts")
        print()
        print(f"  {'language':<{width}} {'script':<12} {'first':>6} {'top 3':>6} {'found':>6}")
        print("  " + "-" * (width + 34))
        for name in sorted(report, key=lambda n: (report[n]["script"], n)):
            d = report[name]
            print(f"  {name:<{width}} {d['script']:<12} "
                  f"{d['first']:>4}/4 {d['top3']:>4}/4 {d['found']:>4}/4")
        total = len(CORPUS) * len(SUBJECTS)
        print("  " + "-" * (width + 34))
        print(f"  {'TOTAL':<{width}} {'':<12} "
              f"{sum(d['first'] for d in report.values()):>4}/{total} "
              f"{sum(d['top3'] for d in report.values()):>4}/{total} "
              f"{sum(d['found'] for d in report.values()):>4}/{total}")
        print()
        for term in sorted(SHARED_TERMS):
            results = archive.query(connection, term, limit=60)
            langs = {r["document"].split("__")[0].split("/")[-1] for r in results}
            expected = {n for n, e in CORPUS.items()
                        if any(term in t for t in e["docs"].values())}
            print(f"  shared term {term!r}: returns {len(langs & expected)} "
                  f"of the {len(expected)} languages that contain it")
