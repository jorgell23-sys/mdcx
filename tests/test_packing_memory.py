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

"""Reading a corpus into a package without holding it.

1.25.0 removed the copies made after the database existed -- its serialised
form, its compressed form, its encrypted form. What remained was the corpus
itself: every document was read into a list before anything was inserted, and
each carries its text and its normalised form, so the list is two copies of the
corpus before the database has been written at all.

Measured by a consumer on a twelfth of their corpus: 25,730 documents, 16.3 GB
resident, extrapolating to some 196 GB on a machine with 96. The cost was not
abstract -- it left them unable to build the single package that would have
removed their other defect, the one where an answer assembled from 66 packages
comes out ordered by the environment variable.
"""
from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive, search as B  # noqa: E402


def _corpus(tmp_path, documents=400, words=90):
    import random

    random.seed(61)
    vocabulary = ("teorema demostracion lema espacio metrico funcion continua "
                  "conjunto abierto cerrado compacto medida integral").split()
    folder = tmp_path if tmp_path.name in ("small", "large") else tmp_path / "md"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(documents):
        blocks = [" ".join(random.choice(vocabulary) for _ in range(words))
                  for _ in range(6)]
        (folder / f"d{i:04}.md").write_text(
            f"# Doc {i}\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return folder


# --- Nothing holds the corpus --------------------------------------------------


def test_documents_are_yielded_one_at_a_time():
    """The reading itself. A list of them is the corpus twice over, and the
    list existed before a single row was written."""
    import inspect

    assert inspect.isgeneratorfunction(B.iter_documents)


def test_the_whole_list_is_still_available_for_whoever_wants_it(tmp_path):
    """`load_documents` is published and other things read it. Deferring is
    for packing, which is where the size is."""
    folder = _corpus(tmp_path, documents=4)

    documents = B.load_documents(folder)

    assert isinstance(documents, list)
    assert len(documents) == 4


def test_packing_does_not_build_a_list_of_the_corpus():
    """Read from the source because the fault is a shape: what matters is that
    nothing between reading and inserting accumulates, whatever it is called."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "archive.py").read_text(encoding="utf-8")
    body = source.split("def _build_database(", 1)[1].split("\ndef ", 1)[0]

    assert "B.iter_documents(folder)" in body
    assert "attachments = [d for d in docs" not in body
    assert "docs = [d for d in docs" not in body


def test_the_peak_does_not_grow_with_the_corpus(tmp_path):
    """The property, measured rather than asserted about the code.

    Not an absolute figure: a small package's peak is dominated by things of
    fixed size -- the compressor's window, the index -- and comparing it to the
    text would measure those. What must not happen is the peak rising in step
    with the material, which is what put the consumer past their machine.

    So two corpora, one four times the other, and the peak is asked to grow by
    much less than four.
    """
    small = _corpus(tmp_path / "small", documents=150)
    large = _corpus(tmp_path / "large", documents=600)

    peaks = []
    for folder, name in ((small, "s"), (large, "l")):
        tracemalloc.start()
        archive.pack(folder, tmp_path / f"{name}.mdcx", "k", preset=1)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

    # SQLite's own memory is not counted -- it is C, and tracemalloc does not
    # see it -- so this bounds the Python side, which is the half that held the
    # corpus.
    growth = peaks[1] / peaks[0]
    assert growth < 2.0, (
        f"four times the corpus raised the peak {growth:.1f} times, which is "
        "the corpus being held again")


# --- And the package is the same package ---------------------------------------


def test_the_counts_survive_the_change(tmp_path):
    """Documents and attachments were separated before anything was numbered,
    to keep the document ids contiguous. They are counted apart now, which is
    the same thing said with two counters."""
    folder = _corpus(tmp_path, documents=6)
    (folder / "attached.pdf.md").write_text(
        "---\nsource_format: pdf\nattachment: true\n---\n\nAn attachment.\n",
        encoding="utf-8")

    written = archive.pack(folder, tmp_path / "c.mdcx", "k")

    assert written["documents"] >= 6
    connection = archive.open_package(tmp_path / "c.mdcx", "k")[0]
    ids = [r[0] for r in connection.execute("SELECT id FROM document ORDER BY id")]
    assert ids == list(range(1, len(ids) + 1)), "document ids are not contiguous"


def test_the_language_is_still_detected(tmp_path):
    """It was read from the first forty documents of a list that no longer
    exists; the same forty are kept as they go past."""
    folder = tmp_path / "md"
    folder.mkdir()
    for i in range(8):
        (folder / f"d{i}.md").write_text(
            f"# Doc {i}\n\nThe quick brown fox jumps over the lazy dog and "
            "does not stop running through the field.\n", encoding="utf-8")

    written = archive.pack(folder, tmp_path / "c.mdcx", "k")

    assert written["language"] == "en"


def test_reading_is_still_timed(tmp_path):
    """Reading is no longer a stage that ends before the next begins, so timing
    it by wrapping the call would report nought and charge the time to
    inserting -- worse than not reporting it."""
    folder = _corpus(tmp_path, documents=20)

    written = archive.pack(folder, tmp_path / "c.mdcx", "k")

    phases = written["seconds_index_by_phase"]
    assert phases["read"] > 0.0, "reading is reported as free"
