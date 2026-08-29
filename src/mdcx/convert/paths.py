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

"""Job planning and portable paths.

No output contains absolute paths. Each document is identified by a
pseudopath beginning with @/, resolved against the output folder.

Example:  @/Received/01 Contract/DOC-001 Scope of Work.md
"""
from __future__ import annotations

import hashlib
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .. import console

PSEUDO_PREFIX = "@"

SUPPORTED = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xlsx",
    ".pptx": "pptx",
    ".epub": "epub",
    ".txt": "text",
    ".md": "text",
    ".csv": "text",
    ".html": "html",
    ".htm": "html",
}

# Formats identified by their first bytes. A file states what it is in its own
# content, and that statement is worth more than its name: a document served
# under the wrong extension is common enough to matter, and treating it by name
# sends it to an engine that cannot read it.
_SIGNATURES = (
    (b"%PDF", "pdf"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole"),   # pre-2007 Office
    (b"PK\x03\x04", "zip"),                              # OOXML, EPUB, ODF
)

# Inside a ZIP, the member that identifies the format.
_ZIP_MARKERS = (
    ("word/", "docx"),
    ("xl/", "xlsx"),
    ("ppt/", "pptx"),
)


def _zip_format(path: Path) -> str | None:
    """The format of a ZIP container, read from what it holds."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            # EPUB states its type in a member named mimetype, which the
            # specification requires to be first and uncompressed.
            if "mimetype" in names:
                try:
                    if archive.read("mimetype").strip() == b"application/epub+zip":
                        return "epub"
                except Exception:  # noqa: BLE001 - a damaged member is not a format
                    pass
            for member in names:
                for prefix, kind in _ZIP_MARKERS:
                    if member.startswith(prefix):
                        return kind
    except Exception:  # noqa: BLE001 - not a readable ZIP, so not one of these
        return None
    return None


def sniff_format(path: Path) -> str | None:
    """The format of a file according to its content, or None when unreadable.

    Returns None rather than guessing when the bytes identify nothing. Plain
    text has no signature to read, so it is left to the extension, which is the
    one case where the name carries information the content does not.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError:
        return None

    for signature, kind in _SIGNATURES:
        if head.startswith(signature):
            if kind == "zip":
                return _zip_format(path)
            if kind == "ole":
                # The OLE container holds Word, Excel or PowerPoint alike, and
                # telling them apart needs a parser. The extension decides.
                return None
            return kind
    return None


def resolve_format(path: Path) -> str | None:
    """The format to convert a file as, preferring what it contains.

    The extension is the fallback, used for text formats, which have no
    signature, and whenever the content identifies nothing.
    """
    return sniff_format(path) or SUPPORTED.get(path.suffix.lower())

def to_pseudopath(rel: Path) -> str:
    """Convert a relative path into a portable pseudopath."""
    return f"{PSEUDO_PREFIX}/" + rel.as_posix()

def from_pseudopath(pseudo: str, root: Path) -> Path:
    """Resolve a pseudopath against a given root."""
    if not pseudo.startswith(PSEUDO_PREFIX + "/"):
        raise ValueError(f"pseudopath invalido: {pseudo!r}")
    return root / pseudo[len(PSEUDO_PREFIX) + 1:]

def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the source file, used to detect changes and skip reconversion."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()

def _normalize(text: str) -> str:
    """Normalise text for comparison: lowercase, accents folded."""
    return unicodedata.normalize("NFC", text)

@dataclass
class Job:
    """A unit of conversion: one source file and its mirrored .md target.

    A long document is split into several jobs, one per chapter. They share the
    same `source` and differ by `page_range`; `parent_target` points at the .md
    index grouping them, so the correspondence with the original stays traceable.
    """

    source: Path                 # absolute path to the original
    rel_source: Path             # path relative to Input/
    rel_target: Path             # path relative to Output/ (ends in .md)
    kind: str                    # engine family: pdf | docx | xlsx | text | ...
    size: int
    digest: str = ""
    meta: dict = field(default_factory=dict)
    page_range: tuple | None = None      # (first, last) 1-based, si es un chapter
    chapter_title: str = ""              # chapter title within the document
    chapter_index: int = 0               # chapter position, starting at 1
    parent_target: Path | None = None    # .md index of the whole document

    @property
    def is_chapter(self) -> bool:
        return self.page_range is not None

    @property
    def pseudopath(self) -> str:
        return to_pseudopath(self.rel_target)

    @property
    def source_pseudopath(self) -> str:
        return to_pseudopath(self.rel_source)

def _walk(root: Path) -> list[Path]:
    """Every file under a folder, skipping what the system will not walk.

    rglob yields lazily, so an error partway through ends the iteration and
    silently truncates the collection -- a long path on Windows, a directory
    without permission, a link that leads nowhere. Walking it here means the
    rest of the tree is still planned, and what was skipped is said out loud
    rather than quietly missing.
    """
    found: list[Path] = []
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            entries = sorted(folder.iterdir())
        except OSError as unreachable:
            console.safe_print(
                f"WARNING: cannot list {folder}, skipped: {unreachable}",
                flush=True)
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    pending.append(entry)
                    continue
            except OSError:
                # Cannot tell what it is. It is still something the folder
                # listed, so it is handed on rather than dropped: the caller
                # interrogates it once more and says out loud that it could not
                # be read. Dropping it here would lose content in silence,
                # which is the failure this walk exists to avoid.
                pass
            found.append(entry)
    return sorted(found)


def plan_jobs(input_root: Path) -> tuple[list[Job], list[Path]]:
    """Walk the input folder and plan one .md per file."""
    jobs: list[Job] = []
    skipped: list[Path] = []
    by_target: dict[str, list[Job]] = {}

    for path in _walk(input_root):
        # One interrogation decides both questions, and its failure is not the
        # same answer as "not a file". is_file() swallows the error and returns
        # False, so an unreadable document would be dropped in silence -- and
        # silence is the wrong outcome for content that exists and cannot be
        # reached. It used to be worse: this call was unguarded, so one such
        # path ended planning and with it the conversion of everything else.
        # Measured elsewhere on the same class of fault, eleven files stopped
        # two stages of a pipeline on every pass for a day.
        try:
            info = path.stat()
        except OSError as unreachable:
            console.safe_print(
                f"WARNING: cannot read {path.name}, skipped: {unreachable}",
                flush=True)
            skipped.append(path)
            continue
        if not stat.S_ISREG(info.st_mode):
            continue

        kind = resolve_format(path)
        if kind is None:
            skipped.append(path)
            continue

        size = info.st_size
        rel_source = path.relative_to(input_root)
        rel_target = rel_source.with_suffix(".md")
        job = Job(
            source=path,
            rel_source=rel_source,
            rel_target=rel_target,
            kind=kind,
            size=size,
        )
        by_target.setdefault(_normalize(rel_target.as_posix()).lower(), []).append(job)
        jobs.append(job)

    for group in by_target.values():
        if len(group) < 2:
            continue
        for job in group:
            tag = job.source.suffix.lower().lstrip(".")
            stem = job.rel_source.stem
            job.rel_target = job.rel_source.with_name(f"{stem}__{tag}.md")

    return jobs, skipped

LANE_GPU = "gpu"   # will need the card: no text to read, so a model must read it
LANE_CPU = "cpu"   # reads its own text; a model here, if reached, runs on the processor

SAMPLE_PAGES = 5


def _pages_to_sample(total: int, how_many: int = SAMPLE_PAGES) -> list[int]:
    """Which pages to look at when deciding whether a document has text.

    Not the first ones. A book opens with a cover, a blank verso and a title
    page: no text, no text, and one line. A sample taken there describes the
    front matter of a document rather than the document, and the thing being
    decided is whether every page of it has to be read by a model.

    Measured on this corpus, that is not a small error. Three volumes of one
    physics textbook were sent to optical recognition on the strength of a title
    page 146 characters long, while the English edition of the same book, whose
    title page runs to 200, was read from its text layer. Both have some 2,700
    characters a page in the body. Fifty-four characters on the third page
    decided between reading 581 pages in seconds and reading them at 85 seconds
    apiece, which is about fourteen hours.

    Spreading the sample through the document costs the same handful of page
    reads and cannot be fooled by what a publisher puts at the front.
    """
    if total <= how_many:
        return list(range(total))
    step = total / (how_many + 1)
    return sorted({min(total - 1, int(step * (i + 1))) for i in range(how_many)})


def inspect_job(job: "Job") -> tuple[str, bool]:
    """The lane for this document, and whether it is likely to reach the card.

    Both answers come out of one sampling pass, because the same few pages
    decide them and opening a document twice to ask two questions about it
    costs about as much as the inspection itself.

    The second answer is the one the lane cannot give. A lane says where a
    document waits; it does not say which engine resolves it. Most documents
    wait in the processor lane and are read from their own text layer, but the
    hybrid engine hands the model every page that announces a table the cheap
    path could not produce -- and the model is on the card. So a document can
    sit in the processor lane and use the card anyway, and sizing the card's
    share by the size of its lane counts the wrong thing.

    That went unnoticed while the lane held everything. Once the lane was
    correctly emptied the count became zero, the share fell to a single
    process, and the 28% of documents that reach the card through the hybrid
    engine queued behind one another: same output, 30% longer.

    What is counted here instead is the table caption. It is the same signal
    the hybrid engine acts on, it is already in text that has been extracted,
    and it costs one match. Ruled pages also reach the model, but finding them
    means walking the objects of the page, which is the expensive half of
    reading it; a caption undercounts and never invents, which is the right
    direction for a number that sizes a lane.
    """
    if job.kind in ("text", "html"):
        return LANE_CPU, False
    if job.kind != "pdf":
        # DOCX/XLSX/PPTX still want the structured engine for their tables and
        # headings, and now get it in either lane, so they no longer compete for
        # the card they were never going to use. The native engine resolves
        # them; layout analysis is the fallback, not the expectation.
        return LANE_CPU, False

    try:
        from . import pdf as _pdf
        from .tables import CAPTION

        doc = _pdf.open_document(job.source)
        try:
            pages = len(doc)
            chosen = _pages_to_sample(pages)
            sample = len(chosen)
            chars = 0
            images = 0
            announces = False
            for i in chosen:
                text = _pdf.page_text(doc[i])
                chars += len(text.strip())
                images += _pdf.count_images(doc[i])
                if not announces and CAPTION.search(text):
                    announces = True
        finally:
            doc.close()
    except Exception:
        return LANE_GPU, True

    cpp = chars / max(1, sample)
    if cpp < 60:
        # Nothing to read. Optical recognition has to read it, and that is the
        # one thing here that genuinely wants the card: measured on this corpus,
        # such documents are 14.8% of the pages and 31% of the processor time.
        return LANE_GPU, True
    if pages <= 2 and images >= 1:
        # drawing or diagram: measured, the heavy engine performs worse
        return LANE_CPU, False
    return LANE_CPU, announces


def classify_lane(job: "Job") -> str:
    """Choose the execution lane from a cheap inspection of the original.

    The lane is about the card, not about the engines. Both lanes may reach the
    structured engine; what the GPU lane holds is the right to use the GPU, and
    it is kept small because the card is what runs out first.

    It used to hold everything, which was right when every document went through
    layout analysis. It no longer does -- most are read from their own text
    layer -- so the general case stopped needing the card and went on queueing
    for it: measured on 22 books, 22 landed in a lane of three processes and
    none in the lane of thirteen. Raising --max-cores did nothing, because it
    was enlarging the empty lane.
    """
    return inspect_job(job)[0]

def estimated_cost(job: "Job") -> float:
    """Estimated conversion cost for this job, in equivalent pages."""
    if job.page_range is not None:
        return float(job.page_range[1] - job.page_range[0] + 1)

    pages = 0
    if job.kind == "pdf":
        from . import pdf as _pdf

        pages = _pdf.count_pages(job.source)
    if not pages:
        return job.size / 100_000
    return float(pages)

def make_chapter_jobs(job: "Job", chapters: list) -> list["Job"]:
    """Expand a long document into one job per chapter."""
    folder = job.rel_target.with_suffix("")   # .../Document/  (no extension)
    out: list["Job"] = []
    for cap in chapters:
        out.append(
            Job(
                source=job.source,
                rel_source=job.rel_source,
                rel_target=folder / f"{cap.slug()}.md",
                kind=job.kind,
                size=job.size,
                page_range=(cap.first_page, cap.last_page),
                chapter_title=cap.title,
                chapter_index=cap.index,
                parent_target=job.rel_target,
            )
        )
    return out
