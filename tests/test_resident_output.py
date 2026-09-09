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

"""What the resident converter writes, which is not the same as what it returns.

The first release of this module answered the report it was written for -- the
models cost sixteen seconds per process and nothing said so -- and introduced a
quieter defect in doing it. It was 2.87x faster, returned well-formed records,
failed at nothing, and wrote a 130-page book as one Markdown where the command
line writes seven chapters and an index. The documentation said the *record*
matched, which is true and reads as "the same as mdcx-convert", which it was
not. Nothing announced the change of granularity, so a consumer who moved here
for the speed lost the unit it retrieves by and found out once its corpus was
already converted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import chapters as chapters_module  # noqa: E402
from mdcx.convert import resident  # noqa: E402


def _chapter(index, first, last, title):
    return chapters_module.Chapter(index=index, title=title, first_page=first,
                                   last_page=last, from_toc=True)


def _record_for(job, output_root, **_kw):
    """What `convert_one` returns for a chapter, in the shape the index reads.

    Faked rather than converted: the question here is what the module does with
    the records, and a real conversion would need a PDF with an outline, which
    is not what is being tested.
    """
    target = output_root / job.rel_target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# {job.chapter_title}\n", encoding="utf-8")
    return {
        "source_pseudopath": job.source_pseudopath,
        "markdown_pseudopath": job.pseudopath,
        "source_name": job.rel_source.name,
        "engine": "nativo",
        "seconds": 0.1,
        "ok": True,
        "chapter": ({"outline": job.chapter_index, "title": job.chapter_title,
                     "pages": [job.page_range[0], job.page_range[1]],
                     "document_pseudopath": "x"}
                    if job.page_range else None),
        "verification": {"coverage": 1.0, "ref_tokens": 100,
                         "missing_tokens": 0},
    }


def _split_into_three(monkeypatch, tmp_path):
    """A document that the planner says has three chapters."""
    monkeypatch.setattr(
        chapters_module, "plan_chapters",
        lambda path, threshold=60: [_chapter(1, 1, 40, "One"),
                                    _chapter(2, 41, 80, "Two"),
                                    _chapter(3, 81, 120, "Three")])
    from mdcx.convert import pdf as pdf_module
    monkeypatch.setattr(pdf_module, "count_pages", lambda path: 120)

    from mdcx.convert import convert as convert_module
    monkeypatch.setattr(convert_module, "convert_one", _record_for)

    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.7\n")
    out = tmp_path / "out"
    out.mkdir()
    return book, out


# --- The output, which is what a consumer keeps --------------------------------


def test_a_long_document_is_written_as_its_chapters(monkeypatch, tmp_path):
    """The retirement condition: the same structure on disk as `mdcx-convert`.
    Matching records was never the thing -- what the consumer keeps is the
    output, and for retrieval the chapter is the unit that gets cited."""
    book, out = _split_into_three(monkeypatch, tmp_path)

    with resident.warm() as convert:
        record = convert(book, out)

    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.md"))
    assert written == ["book.md", "book/01 - One.md", "book/02 - Two.md",
                       "book/03 - Three.md"], written
    assert record["is_document_index"] is True
    assert len(record["chapter_records"]) == 3
    assert record["chapters"] == 3, "the count field was overwritten"


def test_the_index_links_the_chapters(monkeypatch, tmp_path):
    """An index nobody can navigate is a fourth file, not an index."""
    book, out = _split_into_three(monkeypatch, tmp_path)

    with resident.warm() as convert:
        convert(book, out)

    text = (out / "book.md").read_text(encoding="utf-8")
    assert "book/01%20-%20One.md" in text
    assert "pages_total: 120" in text


def test_one_record_a_document_not_a_chapter(monkeypatch, tmp_path):
    """A caller iterating `convert_documents` is counting documents. The
    chapters are reachable, not interleaved into that count."""
    book, out = _split_into_three(monkeypatch, tmp_path)

    records = list(resident.convert_documents([book], out))

    assert len(records) == 1
    assert len(records[0]["chapter_records"]) == 3


def test_not_splitting_is_something_a_caller_can_ask_for(monkeypatch, tmp_path):
    """The difference is a parameter with a documented default, which is the
    other half of the retirement condition."""
    book, out = _split_into_three(monkeypatch, tmp_path)

    with resident.warm() as convert:
        record = convert(book, out, split=False)

    assert "chapter_records" not in record
    assert record["markdown_pseudopath"].endswith("book.md")


# --- What is not split ---------------------------------------------------------


def test_something_that_is_not_a_pdf_is_converted_whole(tmp_path):
    """Chapters are a property of a paginated document. Asking the planner
    about a text file would be asking the wrong question."""
    note = tmp_path / "note.txt"
    note.write_text("# Note\n\nText.\n", encoding="utf-8")

    caps, info = resident.plan_split(resident.job_for(note))

    assert caps == [] and info is None


def test_a_short_document_is_converted_whole(monkeypatch, tmp_path):
    """Under the threshold the planner returns nothing, and the record is the
    single-document one it has always been."""
    monkeypatch.setattr(chapters_module, "plan_chapters",
                        lambda path, threshold=60: [])
    book = tmp_path / "short.pdf"
    book.write_bytes(b"%PDF-1.7\n")

    caps, info = resident.plan_split(resident.job_for(book))

    assert caps == [] and info is None


def test_the_threshold_is_the_one_the_command_line_uses(monkeypatch, tmp_path):
    """Two defaults for the same decision would split differently by which
    entry point was used, which is the defect being closed, again."""
    seen = []
    monkeypatch.setattr(chapters_module, "plan_chapters",
                        lambda path, threshold=60: seen.append(threshold) or [])
    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.7\n")

    resident.plan_split(resident.job_for(book))

    assert seen == [chapters_module.SPLIT_THRESHOLD_PAGES]
