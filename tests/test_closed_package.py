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

"""What can still be done to a package once the material that made it is gone.

Three things a closed package could not do, each of which had assumed the source
folder was still there.

It could not be calibrated. The threshold was writable only by `pack`, and
`pack` walks a folder of documents -- so a package whose material was lost fell
back to a constant nobody measured on it, and would in every future version.
Measured on four such packages, that constant accepts 38 of 48 questions the
corpus answers, against 47 of 48 with a threshold taken from questions.

It could not say which of its words were new. Absence from `df` had two
meanings, and reading it as one turned a novelty measure into a detector of
short function words.

And it had nowhere to keep an object that is not text to search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx import search as B  # noqa: E402
from mdcx import semantic as S  # noqa: E402

QUESTIONS = ["como se transmite el calor por conduccion y radiacion",
             "que es la equivalencia entre masa y energia",
             "como funciona la fotosintesis en las plantas"]


@pytest.fixture
def science(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in (
            ("calor", "El calor se transmite por conduccion, conveccion y radiacion."),
            ("energia", "La equivalencia entre masa y energia se escribe E=mc2."),
            ("foto", "La fotosintesis convierte la luz en energia quimica.")):
        (folder / f"{name}.md").write_text(f"# {name}\n\n{text}\n", encoding="utf-8")
    return folder


# --- Calibrating a package whose material is gone -----------------------------


needs_vectors = pytest.mark.skipif(
    not S.available(), reason="the threshold is a cosine")


@needs_vectors
def test_a_closed_package_can_be_measured_against_questions(science, tmp_path):
    """The whole point: no folder of documents is involved."""
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k", semantic=True)
    for item in science.iterdir():          # the material is gone
        item.unlink()
    science.rmdir()

    result = archive.calibrate(target, "k", QUESTIONS)

    connection, _ = archive.open_package(target, "k")
    assert archive.answerable_at(connection) == result["answerable_at"]
    assert archive.calibrated_from_questions(connection), (
        "measured from questions and not recognised as such, so the wrong "
        "margin would be applied")


@needs_vectors
def test_where_the_measure_came_from_is_kept(science, tmp_path):
    """Measured while packing and measured afterwards are not the same claim.

    One describes the corpus as it was built, the other the corpus as it
    stands. It makes no difference to the margin -- both are thresholds taken
    from questions -- and it belongs in the record.
    """
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k", semantic=True, focus=QUESTIONS)
    connection, _ = archive.open_package(target, "k")
    assert archive.calibrated_from_questions(connection)

    after = tmp_path / "D.mdcx"
    archive.pack(science, after, "k", semantic=True)
    archive.calibrate(after, "k", QUESTIONS)

    assert archive.read_header(target)["answerable_at_from"] == "focus"
    assert archive.read_header(after)["answerable_at_from"] == "focus-after"


@needs_vectors
def test_the_package_is_otherwise_untouched(science, tmp_path):
    """Same documents, same passages, same answers. Only the measure changed."""
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k", semantic=True)
    before, _ = archive.open_package(target, "k")
    was = [r["passage"] for r in archive.query(before, "fotosintesis", limit=5)]

    archive.calibrate(target, "k", QUESTIONS)

    after, header = archive.open_package(target, "k")
    assert header["_intact"], "the package no longer verifies against its header"
    assert [r["passage"] for r in archive.query(after, "fotosintesis", limit=5)] == was


@needs_vectors
def test_a_signed_package_refuses_rather_than_coming_back_unsigned(
        science, tmp_path):
    """Returning an unsigned package where a signed one went in is exactly the
    silent downgrade these reports keep finding."""
    private, _ = archive.generate_signing_key()
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k", semantic=True, signing_key=private)

    with pytest.raises(ValueError, match="signed"):
        archive.calibrate(target, "k", QUESTIONS)

    archive.calibrate(target, "k", QUESTIONS, signing_key=private)
    assert archive.read_header(target)["signature"], "the signature was dropped"


def test_calibrating_without_a_meaning_index_says_which_is_missing(
        science, tmp_path):
    """Two different repairs, so two different messages."""
    target = tmp_path / "W.mdcx"
    archive.pack(science, target, "k")

    with pytest.raises(ValueError, match="meaning index"):
        archive.calibrate(target, "k", QUESTIONS)


# --- Which words are actually new ---------------------------------------------


def test_a_term_the_index_would_never_record_is_not_a_new_term(science, tmp_path):
    """The measurement that made a novelty score a stopword-density detector.

    The terms it called unseen were `no`, `la`, `de`, `en` -- absent because
    they are shorter than the index records, not because the corpus lacks them.
    An absent term takes the maximum idf, so reading absence as novelty makes
    the emptiest words the most informative ones.
    """
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k")
    connection, _ = archive.open_package(target, "k")

    question = "de que manera se transmite el calor en la conduccion"
    df, _, _ = archive.corpus_statistics(connection)
    naive = [t for t in B.tokenize_text(B._normalize(question)) if t not in df]

    assert {"de", "en", "la"} <= set(naive), "premise: the short words look new"
    assert not ({"de", "en", "la"} & set(archive.unknown_terms(connection, question)))


def test_a_genuinely_new_word_is_still_reported(science, tmp_path):
    """Excluding the invisible must not exclude the unfamiliar."""
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k")
    connection, _ = archive.open_package(target, "k")

    unknown = archive.unknown_terms(connection, "la fotosintesis en un cloroplasto")

    assert "cloroplasto" in unknown
    assert "fotosintesis" not in unknown, "a term the corpus has is not new"


def test_the_rule_that_produced_df_travels_with_it(science, tmp_path):
    """Counts without the rule cannot answer the question they are wanted for."""
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k")
    connection, _ = archive.open_package(target, "k")

    known = archive.vocabulary(connection)

    assert known["minimum_term_length"] == archive.MINIMUM_TERM_LENGTH
    assert known["normalized"] is True, (
        "a caller tokenising normally gets GPU where the table holds gpu")
    assert known["terms"] == len(known["df"])


def test_the_index_and_the_rule_do_not_disagree(science, tmp_path):
    """One rule. If indexing and reporting could drift, the answer would be
    wrong in exactly the way this exists to prevent."""
    target = tmp_path / "C.mdcx"
    archive.pack(science, target, "k")
    connection, _ = archive.open_package(target, "k")

    df, _, _ = archive.corpus_statistics(connection)

    assert df, "the fixture indexed nothing"
    assert all(archive.indexable_term(t) for t in df), (
        "the table holds a term the stated rule would have refused")


# --- Something to keep that is not text to search -----------------------------


DATA = "\n\n".join(f"v{i} = {i * 0.31:.4f} {i * 0.77:.4f}" for i in range(300))


@pytest.fixture
def with_artefact(science):
    (science / "certificado.md").write_text(
        f"---\ntitle: certificado\nindexed: false\n---\n\n{DATA}\n",
        encoding="utf-8")
    return science


def test_an_attachment_is_kept_and_not_indexed(with_artefact, tmp_path):
    """Inside the package -- signed, encrypted, one file -- and out of the
    corpus: 2,003 passages of coordinates were 40.6 per cent of a real one."""
    target = tmp_path / "M.mdcx"
    packed = archive.pack(with_artefact, target, "k")

    assert packed["attachments"] == 1
    assert packed["documents"] == 3, "the attachment was counted as a document"

    connection, _ = archive.open_package(target, "k")
    assert not archive.query(connection, "v299", limit=5), "it reached the index"


def test_an_attachment_comes_back_whole(with_artefact, tmp_path):
    """Stored rather than split, so it is returned rather than rebuilt."""
    target = tmp_path / "M.mdcx"
    archive.pack(with_artefact, target, "k")
    connection, _ = archive.open_package(target, "k")

    kept = archive.attachments(connection)

    assert len(kept) == 1
    assert "v299 = 92.6900" in kept[0]["text"]


def test_export_does_not_quietly_lose_it(with_artefact, tmp_path):
    """A folder that looks complete and is not would be the worst outcome."""
    target = tmp_path / "M.mdcx"
    archive.pack(with_artefact, target, "k")

    out = tmp_path / "back"
    result = archive.export(target, "k", out)

    assert result["attachments"] == 1
    assert (out / "certificado.md").is_file()
    assert "v299 = 92.6900" in (out / "certificado.md").read_text(encoding="utf-8")


def test_without_the_marker_nothing_changes(with_artefact, tmp_path):
    """Every package written before this has no such field, and must pack the
    same way it always did."""
    (with_artefact / "certificado.md").write_text(f"# certificado\n\n{DATA}\n",
                                                  encoding="utf-8")
    packed = archive.pack(with_artefact, tmp_path / "N.mdcx", "k")

    assert not packed.get("attachments")
    assert packed["documents"] == 4


def test_pack_says_when_one_document_dominates(with_artefact, tmp_path):
    """A document holding 40 per cent of the passages is what whoever packed it
    would want to see, and it took a SQL query against a package just written."""
    (with_artefact / "certificado.md").write_text(f"# certificado\n\n{DATA}\n",
                                                  encoding="utf-8")
    packed = archive.pack(with_artefact, tmp_path / "N.mdcx", "k")

    biggest = packed["largest_document"]
    assert biggest["name"] == "certificado"
    assert biggest["share"] > 0.9
    assert biggest["passages"] < packed["passages"], "one document is not all of them"
    assert biggest["passages"] == round(biggest["share"] * packed["passages"])
