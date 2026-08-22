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

"""Finding the tables on a page without looking at it.

Reading a page with a vision model costs about a second. Extracting its text
costs three milliseconds. The difference only buys something where the page
holds a table, and in a corpus of books nine pages in ten hold none, so the
expensive reading is paid for almost always and needed almost never.

What makes the cheap path viable is that academic tables announce themselves
twice. They are drawn in the booktabs style: two or three horizontal rules and
no vertical ones. And the author writes a caption, "Table 3.1:", which is a
stronger signal than any geometry because a person put it there on purpose.

So the rules say where the table is, the caption says that it is one, and the
columns are read from the alignment of the text inside that band. The band is
what makes it safe: looking for columns across a whole page invents splits in
the middle of paragraphs.

Detecting a borderless table from text alignment alone is not attempted here,
and that is deliberate. Measured over seven books and 1,353 pages it produced
seven times the table rows a vision model found, marking indexes, lists and
blocks of formulae as tables and eating prose on the way. It is not a matter of
calibration: pdfplumber ships that strategy mature and tuned and scores 8 per
cent precision on the same material, 57 false positives out of 62. That
difficulty is why trained models exist, and a page that reaches the end of this
cascade is better handed to one than guessed at.
"""
from __future__ import annotations

import re

# Two coordinates within this many points are the same line. Also the greatest
# thickness a drawn path may have and still count as a rule rather than a box.
TOLERANCE = 3.0

# A grid whose cells are mostly empty is not a table: it is a page whose lines
# happened to cross. Half of them must hold text.
MIN_OCCUPANCY = 0.5

# "Table 3.1:", "Tabla 2 -", "Tabelle 4.", "Quadro 1:". The number and the
# punctuation are what separate a caption from a sentence mentioning a table.
CAPTION = re.compile(
    r"^\s*(tab(?:le|la|elle|ella)|tabelle|quadro)\s*[\d.]+\s*[.:—-]",
    re.IGNORECASE | re.MULTILINE)


def _rules(page) -> tuple[list[float], list[float]]:
    """The horizontal and vertical rules drawn on the page, by coordinate."""
    import pypdfium2.raw as raw

    width, height = page.get_width(), page.get_height()
    horizontals: list[float] = []
    verticals: list[float] = []
    if not width or not height:
        return horizontals, verticals
    for item in page.get_objects(max_depth=1):
        if item.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        left, bottom, right, top = item.get_bounds()
        w, h = abs(right - left), abs(top - bottom)
        if h <= TOLERANCE and w >= width * 0.15:
            horizontals.append(bottom)
        elif w <= TOLERANCE and h >= height * 0.02:
            verticals.append(left)
    return horizontals, verticals


def _cluster(values: list[float], tolerance: float = TOLERANCE) -> list[float]:
    """Collapse coordinates that are one rule drawn as several paths."""
    if not values:
        return []
    ordered = sorted(values)
    groups = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(g) / len(g) for g in groups]


def _text_in(textpage, left, bottom, right, top) -> str:
    """The text inside a rectangle, assembled by PDFium rather than by us.

    Never rebuild words from character positions. Letters with a descender --
    p, g, y, j, q -- sit lower than their neighbours, so grouping characters by
    the bottom of their box drops them into the next row and tears them out of
    their own word: "OpenStax provides" comes back as "OenStaxrovides", with
    the missing letters on the line below.
    """
    try:
        return (textpage.get_text_bounded(
            left=left, bottom=bottom, right=right, top=top) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _squeeze(text: str) -> str:
    """The text with whitespace removed, for comparing what should be the same.

    Splitting a line into cells and joining them back adds and drops spaces at
    the boundaries, so the comparison has to ignore them: what must not change
    is the letters.
    """
    return "".join(text.split())


def _line_boxes(textpage) -> list[tuple]:
    """The rectangle of every line of text, as PDFium reports them."""
    boxes = []
    try:
        for i in range(textpage.count_rects()):
            boxes.append(textpage.get_rect(i))
    except Exception:  # noqa: BLE001
        pass
    return boxes


def _table_from_grid(textpage, rows: list[float],
                     columns: list[float]) -> str | None:
    """The table when it is drawn whole, or None if that grid is empty."""
    if len(rows) < 3 or len(columns) < 3:
        return None
    rows = sorted(rows, reverse=True)
    columns = sorted(columns)
    n_rows, n_columns = len(rows) - 1, len(columns) - 1
    if n_rows * n_columns > 400:        # an absurd grid is not a table
        return None

    cells, filled = [], 0
    for i in range(n_rows):
        top, bottom = rows[i], rows[i + 1]
        row = []
        for j in range(n_columns):
            value = " ".join(_text_in(textpage, columns[j], bottom,
                                      columns[j + 1], top).split())
            if value:
                filled += 1
            row.append(value)
        cells.append(row)

    if filled < n_rows * n_columns * MIN_OCCUPANCY:
        return None
    out = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * n_columns]
    for row in cells[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _table_band(horizontals: list[float],
                height: float) -> tuple[float, float] | None:
    """The band the horizontal rules enclose, when it could hold a table.

    Academic tables carry no vertical rules at all, because the booktabs style
    draws two or three horizontal ones and leaves the columns to the alignment
    of the text. So there is no grid to read, and the horizontals serve a
    different purpose: they bound where the table is. Restricting the search to
    that band is what keeps column detection from inventing splits in the
    middle of paragraphs.
    """
    if len(horizontals) < 2:
        return None
    top, bottom = max(horizontals), min(horizontals)
    if top - bottom < height * 0.03:    # too thin to hold rows
        return None
    return bottom, top


def _table_in_band(page, textpage, bottom: float, top: float) -> str | None:
    """The table inside a band, with columns read from where text is absent."""
    width = page.get_width()
    boxes = [r for r in _line_boxes(textpage)
             if bottom - 2 <= r[1] and r[3] <= top + 2]
    if not width or len(boxes) < 3:
        return None

    resolution = 2.0
    slots = int(width / resolution) + 1
    occupied = [0] * slots
    for left, _, right, _ in boxes:
        for k in range(max(0, int(left / resolution)),
                       min(slots - 1, int(right / resolution)) + 1):
            occupied[k] += 1

    # A corridor is the vertical gap between two columns. Demanding that no line
    # cross it looks right and is the bottleneck: on the pages a vision model
    # marks as tables, six in fifteen were lost exactly there, because a table
    # with long cells always has some line reaching into the gap. Up to a tenth
    # of the lines may cross it, which inside a band already bounded by drawn
    # rules is not enough to mistake prose for a table.
    #
    # How narrow a corridor may be is measured against the document rather than
    # fixed in points: eight points works at ten-point type and fails at eight,
    # or in a tight table. The median line height stands in for the body size.
    heights = sorted(t - b for _, b, _, t in boxes if t > b)
    body = heights[len(heights) // 2] if heights else 10.0
    narrowest = max(5.0, body * 0.8)
    allowed = max(0, len(boxes) // 10)

    splits, start = [], None
    for k, n in enumerate(occupied):
        if n <= allowed:
            if start is None:
                start = k
        else:
            if start is not None and (k - start) * resolution >= narrowest:
                splits.append(((start + k) / 2) * resolution)
            start = None
    left_edge = min(b[0] for b in boxes)
    right_edge = max(b[2] for b in boxes)
    splits = [s for s in splits if left_edge < s < right_edge]
    if not splits or len(splits) > 10:
        return None

    bounds = [left_edge] + sorted(splits) + [right_edge]
    bands = sorted({(round(b, 1), round(t, 1)) for _, b, _, t in boxes},
                   reverse=True)
    # Lines that overlap are one line: keeping one stops the row repeating.
    distinct: list[tuple[float, float]] = []
    for low, high in bands:
        if distinct and low >= distinct[-1][0] - 1:
            continue
        distinct.append((low, high))
    if len(distinct) < 2:
        return None

    out, filled, seen = [], 0, set()
    for low, high in distinct:
        row = []
        for j in range(len(bounds) - 1):
            value = " ".join(_text_in(textpage, bounds[j], low,
                                      bounds[j + 1], high).split())
            if value:
                filled += 1
            row.append(value)
        if not any(row):
            continue

        # The cells must add up to the line they came from. Where a column
        # boundary falls inside a word the halves land in adjacent cells --
        # "subtracting" arrives as "subtractin" and "g numbers" -- and the row
        # still looks well formed, so nothing downstream would catch it. That
        # is not a table with a flaw, it is prose that was mistaken for one.
        whole = " ".join(_text_in(textpage, bounds[0], low,
                                  bounds[-1], high).split())
        if _squeeze(" ".join(row)) != _squeeze(whole):
            return None

        # Two bands a few points apart can catch the same line twice, and the
        # row then appears repeated for no visible reason.
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        out.append("| " + " | ".join(row) + " |")

    if len(out) < 2:
        return None
    if filled < len(out) * (len(bounds) - 1) * 0.4:
        return None
    out.insert(1, "|" + "---|" * (len(bounds) - 1))
    return "\n".join(out)


def announces_a_table(textpage) -> bool:
    """Whether the page says in its own text that it carries a table.

    In academic material almost every table is labelled, and that is far more
    reliable than any geometry because the author wrote it deliberately. It
    costs one match over text that has already been extracted, and it works in
    both directions: it confirms a page where the geometry is unsure, and it
    withholds confirmation from what carries no label.
    """
    try:
        return bool(CAPTION.search(textpage.get_text_range()))
    except Exception:  # noqa: BLE001
        return False


def table_on_page(page, textpage=None) -> str | None:
    """The table on this page as Markdown, or None if there is none to find.

    Three ways of locating one, most certain first. What none of them does is
    guess from text alignment alone; the module docstring says why that is left
    to a model.
    """
    if textpage is None:
        textpage = page.get_textpage()
    horizontals, verticals = _rules(page)
    rows, columns = _cluster(horizontals), _cluster(verticals)

    # Drawn whole: the grid is there to be read.
    if len(rows) >= 3 and len(columns) >= 3:
        found = _table_from_grid(textpage, rows, columns)
        if found:
            return found

    # Horizontal rules only, which is the common case in this material.
    band = _table_band(rows, page.get_height())
    if band:
        found = _table_in_band(page, textpage, *band)
        if found:
            return found

    # The author says there is a table where the geometry saw too little. With
    # that confirmation a single rule is enough to anchor the search, which on
    # its own it would not be.
    if rows and announces_a_table(textpage):
        height = page.get_height()
        found = _table_in_band(page, textpage,
                               max(0.0, min(rows) - height * 0.35),
                               max(rows) + height * 0.05)
        if found:
            return found
    return None
