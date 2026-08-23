"""Checks that serving several packages does not bury the answer in the others.

Reciprocal rank compares positions, and a position only means something among
lists that are about the same thing. Asking every package and merging one list
per package assumes they are all equally worth reading: the first passage of a
package on an unrelated subject then arrives with the weight of the first
passage of the package that answers, and the merged list converges on equal
shares -- one part in N of it useful, the rest noise, with no error to show for
it.

Nothing here fails in the sense the other tests mean. The call returns exactly
`limit` well-formed passages and exit status says nothing. So these tests ask
about the quality of what comes back, not its shape: given packages on separate
subjects and a query only one of them can answer, most of the answer must come
from that one.

They are skipped without the multilingual extra, because the signal that tells
the packages apart is the cosine of the dense branch, and without it every
package is asked as before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, mcp_server, semantic  # noqa: E402

pytestmark = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")

SUBJECTS = {
    "anatomy": {
        "nervous system": """The autonomic nervous system regulates heart rate and
            digestion. The sympathetic chain ganglia run alongside the vertebral
            column. Preganglionic fibres synapse there before reaching the organ.
            Myelin sheaths increase the speed at which an action potential travels
            along an axon, and glial cells support the neurons around them.""",
        "the cell": """Mitochondria perform respiration and produce adenosine
            triphosphate. Enzymes catalyse reactions by lowering activation energy.
            Membranes are built of phospholipids and decide what enters the cell.""",
        "circulation": """The heart drives blood through arteries and veins.
            Haemoglobin carries oxygen from the lungs to the tissues that need it,
            and returns carrying carbon dioxide to be exhaled.""",
        "the skeleton": """Bone is living tissue, remodelled continuously by cells
            that deposit and resorb mineral. A joint is where two bones meet, and
            cartilage keeps their surfaces from wearing against each other.""",
    },
    "government": {
        "local government": """State and local government administers education and
            public works. A political culture may be moralistic, individualistic or
            traditional. Federalism divides authority between nation and states,
            each of which keeps a constitution and a legislature of its own.""",
        "elections": """An electoral system turns votes into seats. Proportional
            representation allocates them in proportion to the vote, while a
            plurality system gives the seat to whoever leads in a district.""",
        "the courts": """A court interprets the law and settles disputes brought
            before it. Judicial review lets it strike down a statute it finds
            incompatible with the constitution.""",
        "public administration": """A bureaucracy runs on written rules and a
            hierarchy of offices. Civil servants are appointed on examination
            rather than patronage, which is what separates it from spoils.""",
    },
    "manufacturing": {
        "additive manufacturing": """Additive manufacturing builds a part layer by
            layer. Design for additive manufacturing exploits geometries that
            machining cannot produce, such as internal lattices and conformal
            cooling channels that follow the shape of the part.""",
        "fatigue": """A material fails under cyclic load below its tensile strength.
            Fatigue cracks start at a stress concentration and grow each cycle,
            and the S-N curve relates stress amplitude to cycles until failure.""",
        "machining": """A cutting tool removes material to reach a finished shape.
            Feed rate and depth of cut decide both the finish and how quickly the
            tool wears out against the workpiece.""",
        "welding": """Welding joins metals by melting them together at the seam.
            The heat affected zone around the weld has different properties from
            the parent metal, and is where a joint usually fails.""",
    },
}


def build(folder: Path, target: Path, documents: dict, semantic_index: bool) -> Path:
    received = folder / "Received"
    received.mkdir(parents=True, exist_ok=True)
    for name, text in documents.items():
        (received / f"{name}.md").write_text(
            f"---\nsource_format: pdf\n---\n\n# {name}\n\n{text}\n", encoding="utf-8")
    archive.pack(folder, target, "k", semantic=semantic_index)
    return target


def serve(tmp_path, monkeypatch, semantic_index: bool = True) -> None:
    """Three packages on separate subjects, served as one corpus."""
    paths = []
    for topic, documents in SUBJECTS.items():
        paths.append(build(tmp_path / topic, tmp_path / f"{topic}.mdcx",
                           documents, semantic_index))
    monkeypatch.setenv("MDCX_FILE", os.pathsep.join(str(r) for r in paths))
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()


@pytest.fixture
def three_subjects(tmp_path, monkeypatch):
    serve(tmp_path, monkeypatch)
    yield tmp_path
    mcp_server._STATE.clear()


def test_a_query_on_one_subject_is_answered_by_that_package(three_subjects):
    """Every passage should come from the package that holds the subject.

    Before the packages were read for how near they come, this returned one
    passage per package in turn, whatever the query: the share from the package
    that could answer converged on one in N.
    """
    results = mcp_server.search_packages(
        "autonomic nervous system sympathetic chain ganglia", limit=5)
    assert results
    on_topic = [r for r in results if r["package"].startswith("anatomy")]
    assert len(on_topic) > len(results) / 2, (
        f"only {len(on_topic)} of {len(results)} came from the package on topic: "
        f"{[r['package'] for r in results]}")


def test_the_best_passage_is_not_buried(three_subjects):
    """The passage that answers must lead, not sit behind unrelated firsts."""
    first = mcp_server.search_packages(
        "design for additive manufacturing", limit=5)[0]
    assert first["package"].startswith("manufacturing")


def test_a_subject_that_spans_packages_still_reaches_both(three_subjects):
    """The gate must not collapse a query that several packages can answer.

    It is measured against the nearest package rather than against a fixed
    threshold, so it opens when the subject genuinely spans packages and closes
    when it does not.
    """
    results = mcp_server.search_packages(
        "how a system is organised and what makes it fail", limit=6)
    assert len({r["package"] for r in results}) > 1


def test_packages_without_vectors_are_all_asked(tmp_path, monkeypatch):
    """Without the dense branch there is nothing to compare, so nothing is dropped.

    A package that indexes words alone is a supported configuration, not a
    failure, and a query that cannot be measured is not one to filter on.
    """
    serve(tmp_path, monkeypatch, semantic_index=False)
    try:
        packages = mcp_server._open_packages()
        assert mcp_server._packages_worth_asking(packages, "anything at all") == packages
    finally:
        mcp_server._STATE.clear()


def test_a_single_configured_package_is_never_gated(tmp_path, monkeypatch):
    """One package is the whole corpus: there is nothing to be nearer than."""
    build(tmp_path / "only", tmp_path / "only.mdcx", SUBJECTS["anatomy"], True)
    monkeypatch.setenv("MDCX_FILE", str(tmp_path / "only.mdcx"))
    monkeypatch.setenv("MDCX_KEY", "k")
    mcp_server._STATE.clear()
    try:
        results = mcp_server.search_packages("nervous system", limit=3)
        assert results
        assert all("package" not in r for r in results)
    finally:
        mcp_server._STATE.clear()
