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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

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

    source: Path                 # ruta absoluta al original
    rel_source: Path             # ruta relativa a Input/
    rel_target: Path             # ruta relativa a Output/ (termina en .md)
    kind: str                    # familia de engine: pdf | docx | xlsx | text | ...
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

def plan_jobs(input_root: Path) -> tuple[list[Job], list[Path]]:
    """Walk the input folder and plan one .md per file."""
    jobs: list[Job] = []
    skipped: list[Path] = []
    by_target: dict[str, list[Job]] = {}

    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        kind = resolve_format(path)
        if kind is None:
            skipped.append(path)
            continue

        rel_source = path.relative_to(input_root)
        rel_target = rel_source.with_suffix(".md")
        job = Job(
            source=path,
            rel_source=rel_source,
            rel_target=rel_target,
            kind=kind,
            size=path.stat().st_size,
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
    if job.kind in ("text", "html"):
        return LANE_CPU
    if job.kind != "pdf":
        # DOCX/XLSX/PPTX still want the structured engine for their tables and
        # headings, and now get it in either lane, so they no longer compete for
        # the card they were never going to use.
        return LANE_CPU

    try:
        from . import pdf as _pdf

        doc = _pdf.open_document(job.source)
        try:
            pages = len(doc)
            muestra = min(pages, 3)
            chars = sum(len(_pdf.page_text(doc[i]).strip())
                        for i in range(muestra))
            imagenes = sum(_pdf.count_images(doc[i]) for i in range(muestra))
        finally:
            doc.close()
    except Exception:
        return LANE_GPU

    cpp = chars / max(1, muestra)
    if cpp < 60:
        # Nothing to read. Optical recognition has to read it, and that is the
        # one thing here that genuinely wants the card: measured on this corpus,
        # such documents are 14.8% of the pages and 31% of the processor time.
        return LANE_GPU
    if pages <= 2 and imagenes >= 1:
        return LANE_CPU  # drawing or diagram: measured, the heavy engine performs worse
    return LANE_CPU

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
    folder = job.rel_target.with_suffix("")   # .../Documento/  (sin extension)
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
