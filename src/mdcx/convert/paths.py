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
    ".txt": "text",
    ".md": "text",
    ".csv": "text",
    ".html": "html",
    ".htm": "html",
}

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
        ext = path.suffix.lower()
        kind = SUPPORTED.get(ext)
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

LANE_GPU = "gpu"   # requiere modelos de layout/tables/OCR
LANE_CPU = "cpu"   # extractor determinista, sin modelos

def classify_lane(job: "Job") -> str:
    """Choose the execution lane from a cheap inspection of the original."""
    if job.kind in ("text", "html"):
        return LANE_CPU
    if job.kind != "pdf":
        return LANE_GPU  # DOCX/XLSX/PPTX: the structured engine adds tables and headings

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
        return LANE_GPU
    if pages <= 2 and imagenes >= 1:
        return LANE_CPU  # drawing or diagram: measured, the heavy engine performs worse
    return LANE_GPU

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
