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

"""PDF access, built on pypdfium2.

Concentrating PDF access in one module keeps a future change of engine to a
single place. pypdfium2 is used rather than PyMuPDF because the latter is
AGPL-3.0 or commercial, which would determine the licence of any program
depending on it.

pypdfium2 uses the PDF coordinate system, with the origin at the bottom left.
Reading order is therefore reconstructed by descending top edge and then
ascending left edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Block:
    """A text fragment with its position on the page."""
    text: str
    left: float
    top: float

def _open(path: Path):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(str(path))

def count_pages(path: Path) -> int:
    """Number of pages, or zero if the file cannot be read."""
    try:
        doc = _open(path)
    except Exception:  # noqa: BLE001
        return 0
    try:
        return len(doc)
    finally:
        doc.close()

def page_text(page) -> str:
    tp = page.get_textpage()
    try:
        return tp.get_text_range() or ""
    finally:
        tp.close()

def page_blocks(page) -> list[Block]:
    """Text fragments of a page, in reading order."""
    tp = page.get_textpage()
    blocks: list[Block] = []
    try:
        for i in range(tp.count_rects()):
            left, below, right, top = tp.get_rect(i)
            text = (tp.get_text_bounded(left, below, right, top) or "").strip()
            if text:
                blocks.append(Block(text, round(left, 1), round(top, 1)))
    finally:
        tp.close()
    # Highest top coordinate first: in a PDF, further up is a larger value.
    blocks.sort(key=lambda b: (-b.top, b.left))
    return blocks

def count_images(page) -> int:
    import pypdfium2 as pdfium

    try:
        return sum(1 for o in page.get_objects()
                   if o.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE)
    except Exception:  # noqa: BLE001
        return 0

def open_document(path: Path):
    """Open the PDF for iteration. The caller closes it."""
    return _open(path)

def outline(path: Path) -> list[tuple[int, str, int]]:
    """Embedded table of contents as (level, title, page), 1-based."""
    try:
        doc = _open(path)
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[int, str, int]] = []
    try:
        for mark in doc.get_toc():
            try:
                target = mark.get_dest()
                if target is None:
                    continue
                page = target.get_index()
                if page is None:
                    continue
                title = (mark.get_title() or "").strip()
                if title:
                    out.append((mark.level + 1, title, page + 1))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return []
    finally:
        doc.close()
    return out

def extract_pages(source: Path, first: int, last: int, target: Path) -> Path:
    """Write a new PDF containing the given page range, 1-based inclusive."""
    import pypdfium2 as pdfium

    src = _open(source)
    try:
        out = pdfium.PdfDocument.new()
        out.import_pages(src, list(range(first - 1, last)))
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(target))
        out.close()
    finally:
        src.close()
    return target

def outline_for_range(source: Path, first: int, last: int) -> dict[int, list[tuple[int, str]]]:
    """The bookmarks falling in a page range, keyed by page within that range.

    A chapter is converted from a PDF cut out of the book, and cutting pages
    does not carry the bookmarks with them: the extract comes out with none.
    So the headings have to be fetched from the book and translated to where
    the pages ended up.

    They are worth fetching. A chapter of running prose extracted natively has
    no headings at all, and its section titles -- "Osmosis and Osmotic Pressure
    of Solutions" -- are among the best answers a search over the corpus can
    return. The author already wrote them into the document as a table of
    contents, with the hierarchy included, so there is nothing to infer.
    """
    found: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in outline(source):
        if first <= page <= last:
            found.setdefault(page - first + 1, []).append((level, title))
    return found

def extract_page_list(source: Path, pages: list[int], target: Path) -> Path:
    """Write a new PDF holding the given pages, zero-based, in order.

    Gathering scattered pages into one document is what lets an engine that
    charges per document read them in a single pass. Converting them one at a
    time costs that charge once per page: measured on four pages of a textbook,
    separately they took longer than converting the whole forty-page extract.
    """
    import pypdfium2 as pdfium

    src = _open(source)
    try:
        out = pdfium.PdfDocument.new()
        out.import_pages(src, sorted(set(pages)))
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(target))
        out.close()
    finally:
        src.close()
    return target

LINE_TOLERANCE = 2.0

PARAGRAPH_GAP = 14.0

def page_paragraphs(page) -> list[str]:
    """Page text grouped into lines and paragraphs, in reading order."""
    blocks = page_blocks(page)
    if not blocks:
        return []

    lines: list[tuple[float, str]] = []
    current: list[Block] = []
    for b in blocks:
        if current and abs(current[-1].top - b.top) <= LINE_TOLERANCE:
            current.append(b)
            continue
        if current:
            lines.append((current[0].top,
                           " ".join(x.text for x in sorted(current, key=lambda y: y.left))))
        current = [b]
    if current:
        lines.append((current[0].top,
                       " ".join(x.text for x in sorted(current, key=lambda y: y.left))))

    paragraphs: list[str] = []
    accumulated: list[str] = []
    previous: float | None = None
    for top, text in lines:
        if previous is not None and (previous - top) > PARAGRAPH_GAP:
            if accumulated:
                paragraphs.append(" ".join(accumulated))
            accumulated = []
        accumulated.append(text)
        previous = top
    if accumulated:
        paragraphs.append(" ".join(accumulated))
    return paragraphs

def page_paragraphs_fast(page) -> list[str]:
    """Paragraphs from the plain page text, without inspecting rectangles."""
    text = page_text(page)
    if not text.strip():
        return []
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        l = line.strip()
        if l:
            current.append(l)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs
