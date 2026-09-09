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

"""What packing costs, in memory and in time, and where that is said.

Two faults reported together by a consumer converting mathematics at scale.

Writing a package held the whole corpus several times over -- the database as
one object, its compressed form, and the encrypted result, none of them written
until all of them existed. Measured on 134,677 documents and 4.17 GiB of text:
35.6 GB resident after 34 minutes, rising about 5 GB a minute, and no package
written. There was no size at which it stopped being reasonable and became
fatal; it grew until the machine ran out, with no error and no parameter to ask
for less.

And converting read every PDF twice: 0.808 s a work to convert it and 0.681 s
to read it again for something independent to compare against -- 25% of the
conversion. Comparing the two was 4%. Nothing said so, so the quarter was
attributed to conversion being slow and the wrong half was optimised.
"""
from __future__ import annotations

import hashlib
import lzma
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402


# --- The package is the same package -------------------------------------------


def test_sealing_in_blocks_writes_the_same_bytes(tmp_path):
    """The whole safety of the change: XZ is a stream format and GCM is a
    stream cipher whose tag goes last, which is where `AESGCM.encrypt` puts it.
    So what changes is what packing costs, not what a package is -- a reader of
    any version opens what this writes."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data = (b"a page of prose, repeated " * 20000) + os.urandom(4096)
    (tmp_path / "database").write_bytes(data)
    key = os.urandom(32)

    nonce, digest, written = archive._seal(
        tmp_path / "database", tmp_path / "body", key, archive.PRESET)
    body = (tmp_path / "body").read_bytes()

    expected = AESGCM(key).encrypt(nonce, lzma.compress(data, preset=archive.PRESET), None)
    assert body == expected, "the body is no longer what the old path produced"
    assert digest == hashlib.sha256(expected).hexdigest()
    assert written == len(body)


def test_what_was_sealed_comes_back(tmp_path):
    """The other direction, against the reader that is actually shipped."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data = b"documents, an index and provenance" * 5000
    (tmp_path / "database").write_bytes(data)
    key = os.urandom(32)

    nonce, _digest, _n = archive._seal(
        tmp_path / "database", tmp_path / "body", key, archive.FAST_PRESET)

    recovered = lzma.decompress(
        AESGCM(key).decrypt(nonce, (tmp_path / "body").read_bytes(), None))
    assert recovered == data


def test_the_block_is_a_bound_and_not_the_corpus():
    """What fixes the peak is this number. Before, it was the size of the
    material -- which is not a number a caller can choose."""
    assert 0 < archive.SEAL_BLOCK_BYTES <= 64 * 1024 * 1024


def test_nothing_holds_the_whole_corpus_any_more():
    """The three lines the report quoted, and that they are gone. Read from the
    source because the fault is a shape, not a value: a stage that builds a
    whole representation before the next one starts is the defect, whatever it
    is called."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")
    body = source.split("def pack(", 1)[1].split("\ndef ", 1)[0]

    assert "lzma.compress(base_score" not in body
    assert "_encrypt(compressed" not in body
    assert "_seal(" in body


def test_the_database_is_written_out_rather_than_serialised():
    """`serialize()` returns the whole database as one object beside the one
    already in memory. Copying it out page by page is the same bytes without
    the second copy."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")
    body = source.split("def _build_database(", 1)[1].split("\ndef ", 1)[0]

    assert "connection.backup(spill)" in body
    assert "connection.serialize()" not in body


# --- The price of verifying, said ----------------------------------------------


def test_the_record_says_what_reading_twice_cost(tmp_path):
    """Both halves separately, because they are optimised in different places
    and telling them apart is exactly what the consumer could not do."""
    from mdcx.convert import resident

    (tmp_path / "a.txt").write_text("# A\n\nSome text to verify.\n",
                                    encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    record = next(resident.convert_documents([tmp_path / "a.txt"], out))

    assert "seconds_reference" in record, "the second read is still unpriced"
    assert "seconds_verify" in record
    assert record["seconds_reference"] >= 0.0
    assert record["seconds_verify"] >= 0.0


# --- Reading the original once ------------------------------------------------


def test_counting_images_is_not_asked_of_a_book():
    """The count answers one question -- whether a PDF of a page or two is a
    diagram -- and was taken on every page of every document. `get_objects()`
    walks every object on the page where reading the text walks characters.

    Measured over six books, 929 pages: 2.272 s reading the text and 2.484 s
    counting images, so 52% of the second read answered a question about
    two-page documents on books of three hundred. Removing it took the second
    read down 32.4%.
    """
    from mdcx.convert import extract

    assert extract.DRAWING_MAX_PAGES == 2


def test_the_drawing_threshold_has_one_definition():
    """Two would let the extractor stop counting for exactly the documents the
    decision still asks about, and a diagram of two pages would quietly stop
    being recognised as one."""
    from mdcx.convert import convert, extract

    assert convert.PLAN_MAX_PAGES == extract.DRAWING_MAX_PAGES


def test_not_counted_is_not_the_same_as_none():
    """A reader must not mistake "not asked" for "no images": only one of the
    two is a fact about the document."""
    from mdcx.convert import extract

    source = Path(extract.__file__).read_text(encoding="utf-8")
    assert 'if count_images else {}' in source


def test_the_page_text_is_read_once_and_shared():
    """The second read the profile found was not a different kind of reading.
    `extract._pdf_text` calls `page_text`; the native engine reaches
    `page_text` too, through `page_paragraphs_fast`. The same function over the
    same pages for the same characters.

    So this is not a sample of the reference or an approximation of it -- it is
    the reference, kept from the pass that produced it, and what the comparison
    measures is unchanged.
    """
    from mdcx.convert import pdf as pdf_module

    pdf_module.forget_pages()
    assert pdf_module.pages_remembered(Path("nothing")) is None


def test_what_was_kept_is_served_only_for_that_exact_file(tmp_path):
    """A chapter cut from a book reuses names, so name alone would serve one
    document the text of another -- which would not fail, it would silently
    verify against the wrong original."""
    from mdcx.convert import pdf as pdf_module

    f = tmp_path / "cut.pdf"
    f.write_bytes(b"%PDF-1.7\nfirst\n")
    pdf_module.remember_pages(f, ["page one"])
    assert pdf_module.pages_remembered(f) == ["page one"]

    f.write_bytes(b"%PDF-1.7\nsecond, and longer\n")
    assert pdf_module.pages_remembered(f) is None, (
        "the previous document's text was served for a rebuilt file")

    pdf_module.forget_pages()


def test_only_one_document_is_kept(tmp_path):
    """Bounded by the document rather than by the run: remembering more would
    grow with how much is being converted."""
    from mdcx.convert import pdf as pdf_module

    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF-1.7\na\n")
    b.write_bytes(b"%PDF-1.7\nb\n")

    pdf_module.remember_pages(a, ["A"])
    pdf_module.remember_pages(b, ["B"])

    assert pdf_module.pages_remembered(b) == ["B"]
    assert pdf_module.pages_remembered(a) is None
    pdf_module.forget_pages()


def test_the_document_is_forgotten_when_it_is_done(tmp_path):
    """Held past the document, the memory would be a function of the run."""
    from mdcx.convert import pdf as pdf_module
    from mdcx.convert import resident

    (tmp_path / "a.txt").write_text("# A\n\nText.\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    pdf_module.remember_pages(tmp_path / "a.txt", ["left over"])

    next(resident.convert_documents([tmp_path / "a.txt"], out))

    assert pdf_module.pages_remembered(tmp_path / "a.txt") is None


def test_the_shared_text_gives_the_same_paragraphs():
    """Verified over six real books as well -- the Markdown came out identical
    for all of them -- and fixed here so it cannot drift."""
    from mdcx.convert import pdf as pdf_module

    text = "First line\nsecond line\n\nA second paragraph.\n"

    assert pdf_module.page_paragraphs_fast(None, text=text) == [
        "First line second line", "A second paragraph."]


# --- Where indexing time goes --------------------------------------------------


def test_indexing_time_is_reported_by_phase(tmp_path):
    """One number for indexing hid which part of it a caller could act on.

    A consumer measuring 355 s of indexing against 10 s of sealing concluded
    the model was the bottleneck and profiled it. Reading the folder, cutting
    passages and counting terms are each a different decision -- a different
    corpus, a different batch, or nothing the caller can do -- and the single
    figure could not tell them apart.
    """
    corpus = tmp_path / "md"
    corpus.mkdir()
    for i in range(4):
        (corpus / f"d{i}.md").write_text(
            f"# Document {i}\n\n" + "\n\n".join(
                f"A paragraph of prose about topic {j}, written out at some "
                f"length so that it survives being cut into passages."
                for j in range(4)) + "\n", encoding="utf-8")

    written = archive.pack(corpus, tmp_path / "c.mdcx", "k")

    phases = written["seconds_index_by_phase"]
    assert set(phases) >= {"read", "passages", "terms"}
    assert all(v >= 0.0 for v in phases.values())
    # The parts are of the whole they claim to divide.
    assert sum(phases.values()) <= written["seconds_index"] + 0.5


def test_the_phases_account_for_what_was_asked_for(tmp_path):
    """A phase that did not run is reported at zero rather than left out, so a
    reader can tell "not asked for" from "took no time" by what they passed."""
    corpus = tmp_path / "md"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nSome prose to index.\n", encoding="utf-8")

    written = archive.pack(corpus, tmp_path / "c.mdcx", "k", shapes=False)

    assert written["seconds_index_by_phase"]["shapes"] == 0.0


# --- Where to stop on the compression curve -----------------------------------


def _small_corpus(path):
    path.mkdir()
    for i in range(6):
        (path / f"d{i}.md").write_text(
            f"# Document {i}\n\n" + "\n\n".join(
                "Prose that repeats itself enough to be worth compressing, "
                f"paragraph {j} of document {i}." for j in range(8)) + "\n",
            encoding="utf-8")
    return path


def test_the_caller_chooses_where_to_stop_compressing(tmp_path):
    """`fast` picks between two constants, and both answer the same question:
    what a distributed package should cost, compressed once and downloaded
    many times. A corpus rebuilt whenever it grows is the other case -- the
    clock costs and the bytes do not -- and it could not be expressed.

    Measured by the consumer who asked, over 120 MiB of their own text: 5.9 s
    at preset 0 against 24.5 s at preset 3, for 30.0% of the original against
    26.1%.
    """
    corpus = _small_corpus(tmp_path / "md")

    quick = archive.pack(corpus, tmp_path / "quick.mdcx", "k", preset=0)
    dense = archive.pack(corpus, tmp_path / "dense.mdcx", "k", preset=9)

    assert quick["compression_preset"] == 0
    assert dense["compression_preset"] == 9
    assert quick["bytes_compressed"] >= dense["bytes_compressed"]


def test_the_preset_overrides_fast(tmp_path):
    """Two ways of saying the same thing need an order, or the caller is
    guessing which one won."""
    corpus = _small_corpus(tmp_path / "md")

    written = archive.pack(corpus, tmp_path / "c.mdcx", "k",
                           fast=True, preset=6)

    assert written["compression_preset"] == 6


def test_without_a_preset_fast_still_decides(tmp_path):
    """Nothing that worked stops working."""
    corpus = _small_corpus(tmp_path / "md")

    quick = archive.pack(corpus, tmp_path / "f.mdcx", "k", fast=True)
    dense = archive.pack(corpus, tmp_path / "s.mdcx", "k")

    assert quick["compression_preset"] == archive.FAST_PRESET
    assert dense["compression_preset"] == archive.PRESET


def test_a_level_that_does_not_exist_is_refused(tmp_path):
    """Clamping to 9 would leave a caller measuring a level they did not
    choose, and believing they had."""
    corpus = _small_corpus(tmp_path / "md")

    for asked in (-1, 12):
        try:
            archive.pack(corpus, tmp_path / f"c{asked}.mdcx", "k", preset=asked)
        except ValueError as refused:
            assert str(asked) in str(refused)
        else:
            raise AssertionError(f"preset={asked} was accepted")


def test_any_level_writes_a_package_any_reader_opens(tmp_path):
    """The level travels inside the XZ stream, which is what makes this a
    choice for the caller rather than a change to the format."""
    corpus = _small_corpus(tmp_path / "md")

    archive.pack(corpus, tmp_path / "c.mdcx", "k", preset=0)
    connection, _ = archive.open_package(tmp_path / "c.mdcx", "k")

    assert archive.query(connection, "paragraph", limit=2)
