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

"""Splitting of long documents into chapters.

A document of several hundred pages occupies one worker for a long time
while the rest of the queue waits. Split by its embedded table of contents,
the sections are distributed across workers and the output remains readable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SPLIT_THRESHOLD_PAGES = 60

TARGET_CHAPTER_PAGES = 30

FALLBACK_BLOCK_PAGES = 25

@dataclass
class Chapter:
    """A page range of a document, with its corresponding title."""

    index: int          # position within the document, starting at 1
    title: str          # outline title, or a page label when no outline exists
    first_page: int     # 1-based, inclusivo
    last_page: int      # 1-based, inclusivo
    from_toc: bool      # True when the split comes from the document outline

    @property
    def pages(self) -> int:
        return self.last_page - self.first_page + 1

    def slug(self) -> str:
        """Readable, stable file name for the chapter."""
        clean = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", self.title).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) > 70:
            clean = clean[:70].rstrip()
        if not clean:
            clean = f"Pages {self.first_page}-{self.last_page}"
        return f"{self.index:02d} - {clean}"

def _toc_chapters(toc: list, page_count: int) -> list[Chapter]:
    """Build chapters from the embedded table of contents."""
    if not toc:
        return []

    niveles = sorted({lv for lv, _t, _p in toc})
    chosen: list[tuple[str, int]] = []

    for level in niveles:
        entries = [(title, page) for lv, title, page in toc
                    if lv == level and 1 <= page <= page_count]
        if not entries:
            continue
        # Discard out-of-order entries: a malformed outline can point backwards.
        # negativos y chapters vacios.
        entries = sorted(entries, key=lambda e: e[1])
        average = page_count / len(entries)
        chosen = entries
        if average <= TARGET_CHAPTER_PAGES * 1.6:
            break  # this level already yields ranges of a reasonable size

    if not chosen:
        return []

    chapters: list[Chapter] = []
    if chosen[0][1] > 1:
        chapters.append(Chapter(1, "Preliminares", 1, chosen[0][1] - 1, True))

    for i, (title, page) in enumerate(chosen):
        end = chosen[i + 1][1] - 1 if i + 1 < len(chosen) else page_count
        if end < page:
            continue  # two entries on one page: the next absorbs the range
        chapters.append(Chapter(len(chapters) + 1, title.strip(), page, end, True))

    return [c for c in chapters if c.pages > 0]

def _block_chapters(page_count: int, block: int = FALLBACK_BLOCK_PAGES) -> list[Chapter]:
    """Fixed-size ranges, for documents without a table of contents."""
    chapters: list[Chapter] = []
    start = 1
    while start <= page_count:
        end = min(start + block - 1, page_count)
        chapters.append(
            Chapter(len(chapters) + 1, f"Pages {start} a {end}", start, end, False)
        )
        start = end + 1
    return chapters

def plan_chapters(pdf_path: Path, threshold: int = SPLIT_THRESHOLD_PAGES) -> list[Chapter]:
    """Return the chapters of a PDF, or an empty list if splitting is not warranted."""
    from . import pdf as _pdf

    page_count = _pdf.count_pages(pdf_path)
    if page_count < threshold:
        return []
    toc = _pdf.outline(pdf_path)

    chapters = _toc_chapters(toc, page_count)

    if chapters:
        refined: list[Chapter] = []
        for cap in chapters:
            if cap.pages > TARGET_CHAPTER_PAGES * 2:
                start = cap.first_page
                part = 1
                while start <= cap.last_page:
                    end = min(start + TARGET_CHAPTER_PAGES - 1, cap.last_page)
                    title = cap.title if part == 1 else f"{cap.title} (cont. {part})"
                    refined.append(
                        Chapter(len(refined) + 1, title, start, end, cap.from_toc)
                    )
                    start = end + 1
                    part += 1
            else:
                refined.append(
                    Chapter(len(refined) + 1, cap.title, cap.first_page, cap.last_page,
                            cap.from_toc)
                )
        return refined

    return _block_chapters(page_count)

def extract_range(source: Path, first_page: int, last_page: int, target: Path) -> Path:
    """Write a new PDF containing the given page range (1-based, inclusive)."""
    from . import pdf as _pdf

    return _pdf.extract_pages(source, first_page, last_page, target)
