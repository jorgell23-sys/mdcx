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

"""Converting one document at a time without paying for the models each time.

The models are loaded per process, not per call. Measured over pages taken from
real books and converted inside one interpreter: the first call costs 12.97
seconds and the second 0.48, and fitting the warm calls gives 0.252 s a page
with no per-call fixed cost worth the name. Importing torch and docling adds
another 3.7. So a process costs about sixteen seconds before it converts
anything, and nothing said so.

That is the whole of it for anyone orchestrating their own queue -- one process
per document, which is the natural way to spread work and isolate failures.
Measured on 24 books at 14.90 s each, some sixteen of those seconds are startup:
36 per cent. Over a harvest of 886,086 works it is 3,900 machine-hours spent
loading the same weights again.

Until now the only way to amortise it was to hand a large folder to one
`mdcx.cli` run, which means giving up the order, the per-document reaction and
the failure isolation that a queue is for. This is the other way: keep the
process, keep the control.

    from mdcx.convert import resident

    with resident.warm() as convert:          # pays the ~16 s once
        for pdf in my_queue:
            record = convert(pdf, output_root)
            ...                               # decide, retry, stop, reorder

Measured by a consumer with its own queue, on eight books through the native
path: 4.85 s a work invoking `mdcx.cli` once per document against 1.69 s here,
which is 2.87x and is exactly the startup no longer being paid per work.

**What it writes.** The same files `mdcx-convert` writes, chapters included: a
long PDF becomes one Markdown per chapter in a folder of the document's name,
plus an index Markdown linking them. This is not a detail of presentation --
for anyone converting in order to retrieve passages, the chapter is the unit
that gets found and cited, and a book as a single 130-page file is one place
instead of seven. The first release of this module did not split, said only
that the *record* matched `mdcx-convert`, and so lost the granularity of
anyone who moved here for the speed without noticing. `split=False` asks for
the unsplit form deliberately.

What it does not do is run anything in parallel. One process converts one
document at a time, and several of these are several processes -- each paying
its own startup once and amortising it over everything it is given. Where the
whole corpus is known in advance and none of that control is wanted,
`mdcx-convert` over a folder is still the shorter way.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from .paths import Job, file_digest, make_chapter_jobs, to_pseudopath


def cost_of_starting() -> float:
    """Seconds spent making the engines usable, paid once per process.

    Reported rather than assumed, because the number is what tells a caller
    whether a batch is worth amortising over. It is measured on the first call
    and remembered, so asking twice is free and gives the same answer.
    """
    from . import engines

    if _STARTUP[0] is None:
        started = time.perf_counter()
        # Importing torch and the structured engine, which is the part that can
        # be paid up front. The model weights themselves load on the first
        # document that needs them -- some 12.5 further seconds -- and there is
        # no way to ask for them without a document to convert, so this number
        # is the smaller half and says so.
        engines.gpu_available()
        _STARTUP[0] = time.perf_counter() - started
    return _STARTUP[0]


_STARTUP: list = [None]


def job_for(source: Path, input_root: Path | None = None) -> Job:
    """One job for one file, the way `plan_jobs` would have made it.

    Built here so that a caller with its own queue does not have to reproduce
    the shape of a Job -- which is internal, and would then be two definitions
    of the same thing drifting apart.
    """
    source = Path(source).resolve()
    root = Path(input_root).resolve() if input_root else source.parent
    try:
        relative = source.relative_to(root)
    except ValueError:
        relative = Path(source.name)
    try:
        size = source.stat().st_size
    except OSError:
        size = 0
    return Job(
        source=source,
        rel_source=relative,
        rel_target=relative.with_suffix(".md"),
        kind=_kind_of(source),
        size=size,
        digest=file_digest(source),
    )


def _kind_of(source: Path) -> str:
    """The engine family, from the extension, as planning decides it."""
    suffix = source.suffix.lower().lstrip(".")
    if suffix in ("pdf", "docx", "xlsx", "pptx", "html", "htm"):
        return "html" if suffix in ("html", "htm") else suffix
    if suffix in ("txt", "md", "csv"):
        return "text"
    return suffix or "text"


def plan_split(job: Job, threshold: int | None = None) -> tuple[list, dict | None]:
    """The chapters of this document and what an index of them would need.

    The same decision `mdcx.cli` makes between planning and converting, in one
    place so the two cannot answer it differently. A document under the
    threshold, one without a usable outline, and anything that is not a PDF all
    come back with no chapters, which is the caller's signal to convert it
    whole.
    """
    from . import chapters as _chapters
    from . import pdf as _pdf

    if job.kind != "pdf":
        return [], None
    threshold = (_chapters.SPLIT_THRESHOLD_PAGES if threshold is None
                 else threshold)
    caps = _chapters.plan_chapters(job.source, threshold)
    if len(caps) < 2:
        return [], None
    return caps, {
        "document": job.rel_source.name,
        "pseudopath_index": to_pseudopath(job.rel_target),
        "chapters_folder": to_pseudopath(job.rel_target.with_suffix("")),
        "chapters": len(caps),
        # What the original had, beside what the chapters cover. A document
        # split into chapters that do not span it is a truncated conversion,
        # and with only the covered count there is no way to tell.
        "pages_total": _pdf.count_pages(job.source),
        "from_pdf_outline": caps[0].from_toc,
        "rel_target": job.rel_target,
        "rel_source": job.rel_source,
        "children": [],
    }


@contextmanager
def warm(report=None):
    """Hold the engines for as long as the block runs, and hand back a converter.

    The converter takes a path and an output folder and returns the record for
    that document -- the same record `mdcx-convert` writes, so whatever reads
    one reads the other. Where the document was split, the record returned is
    the index record -- `is_document_index` -- and the chapter records are
    under `record["chapter_records"]`, which is the same pair of things
    `mdcx-convert` puts in its manifest.

    Errors are not swallowed here: a caller keeping its own queue is exactly
    the caller that wants to decide what to do about a document that failed.

    `report` is called once with the seconds the startup cost, so a log can say
    it rather than a reader having to measure it.
    """
    from .convert import convert_one

    spent = cost_of_starting()
    if report is not None:
        report(spent)

    def convert(source, output_root, *, input_root=None, use_docling=True,
                save_lossless=True, compact=True, split=True,
                split_threshold=None) -> dict:
        from . import index as _index

        output_root = Path(output_root)
        job = job_for(source, input_root)
        caps, info = (plan_split(job, split_threshold) if split else ([], None))
        if not caps:
            return convert_one(job, output_root, use_docling=use_docling,
                               save_lossless=save_lossless, compact=compact)

        records = []
        for chapter_job in make_chapter_jobs(job, caps):
            record = convert_one(chapter_job, output_root,
                                 use_docling=use_docling,
                                 save_lossless=save_lossless, compact=compact)
            record["is_chapter"] = True
            records.append(record)
        document = _index.write_document_index(info, records, output_root)
        # Not under "chapters": the index record already uses that for the
        # count, which the manifest reads. Overwriting it with the list would
        # be this module quietly changing a field of a shared format.
        document["chapter_records"] = records
        return document

    yield convert


def convert_documents(sources, output_root, *, input_root=None,
                      use_docling=True, save_lossless=True, compact=True,
                      split=True, split_threshold=None):
    """Convert each of these documents in this process, yielding one record each.

    The short form of `warm`, for a caller that already has the list. It is a
    generator on purpose: the record for the first document arrives before the
    second is started, so a queue can act on it.

    One record a document, not a chapter: a caller iterating this is counting
    documents. The chapters of a split document are under
    `record["chapter_records"]`.
    """
    with warm() as convert:
        for source in sources:
            yield convert(source, output_root, input_root=input_root,
                          use_docling=use_docling,
                          save_lossless=save_lossless, compact=compact,
                          split=split, split_threshold=split_threshold)
