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


def _slot(value: float, bounds: list[float]) -> int | None:
    """Which interval of a sorted list of boundaries a value falls in."""
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    if bounds and value == bounds[-1]:
        return len(bounds) - 2      # the far edge belongs to the last interval
    return None


def _grid_cells(textpage, rows: list[float],
                columns: list[float]) -> list[list[str]]:
    """The cells of a drawn grid, each character counted in exactly one.

    Cells cannot be cut out as rectangles here. Neighbouring cells share the
    boundary between them, and PDFium returns every character whose box
    *intersects* the rectangle asked for, so a character straddling a boundary
    comes back in both -- and the row still has the right number of columns
    with every cell filled, which is why nothing downstream notices. Measured
    over 292 rows of drawn tables, 64 of them carried letters twice: a
    concentration written 2.00x10-6 arrived as a column holding "2" and another
    holding "2.00 10-6".

    Shrinking each rectangle does not fix it. A character box can reach several
    points into the next cell, and pulling the edge back far enough to exclude
    it starts cutting characters that belong: at two points of margin the rows
    with duplicated letters fall from 64 to 3, and 43 rows lose letters they
    should have kept.

    A character has one centre, and that centre falls in one column. Walking
    the characters and placing each by its centre counts every one exactly
    once -- measured on the same 292 rows, nothing duplicated and nothing lost.

    This does not contradict what `_text_in` warns about rebuilding words from
    character positions. That warning is about grouping by *row*, where a
    descender drops a letter into the line below. Here the rows are already
    fixed by the drawn rules and only the column is in question, and reading
    order survives because the characters are walked in index order.
    """
    cells = [[[] for _ in columns[:-1]] for _ in rows[:-1]]
    rows_ascending = sorted(rows)
    try:
        total = textpage.count_chars()
    except Exception:  # noqa: BLE001
        return [[""] * (len(columns) - 1) for _ in rows[:-1]]

    for index in range(total):
        try:
            left, low, right, high = textpage.get_charbox(index, loose=False)
        except Exception:  # noqa: BLE001
            continue
        middle_y = (low + high) / 2
        middle_x = (left + right) / 2
        if high < rows[-1] or low > rows[0]:
            continue                        # above or below the grid entirely
        if right <= columns[0] or left >= columns[-1]:
            continue                        # outside it to the left or right

        row_at = _slot(middle_y, rows_ascending)
        column_at = _slot(middle_x, columns)
        if row_at is None or column_at is None:
            # The centre falls outside the grid although the character touches
            # it: the outer rule cuts through the glyph. It belongs to the cell
            # at that edge, which is where the rest of its word is -- the
            # alternative is dropping the first letter of the line, which is
            # what this cost before: "um Numbers" for "Quantum Numbers".
            if row_at is None:
                row_at = 0 if middle_y < rows[-1] else len(rows) - 2
            if column_at is None:
                column_at = 0 if middle_x < columns[0] else len(columns) - 2
        try:
            cells[len(rows) - 2 - row_at][column_at].append(
                textpage.get_text_range(index, 1))
        except Exception:  # noqa: BLE001
            pass
    return [[" ".join("".join(cell).split()) for cell in row] for row in cells]


def cells_by_word(textpage, rows: list[float], columns: list[float]) -> list[list[str]]:
    """The cells of a grid, placing whole words rather than single characters.

    For a grid read off a rendering rather than off the drawn rules, the
    boundaries are approximate: they come from pixels and land wherever the
    model put them, which is often a point or two inside a word. Placing each
    character by its own centre then splits that word across two cells, and
    the row is well formed and wrong.

    A word has one centre too. Grouping the characters into words first, and
    placing each word whole, keeps the model's geometry -- which is right about
    where the columns are -- while refusing to cut anywhere the text does not
    already have a gap.
    """
    words = _words_of(textpage)
    rows_ascending = sorted(rows)
    cells = [[[] for _ in columns[:-1]] for _ in rows[:-1]]
    for text, left, bottom, right, top in words:
        if not text.strip():
            continue
        middle_y = (bottom + top) / 2
        middle_x = (left + right) / 2
        if top < rows[-1] or bottom > rows[0]:
            continue
        if right <= columns[0] or left >= columns[-1]:
            continue
        row_at = _slot(middle_y, rows_ascending)
        column_at = _slot(middle_x, columns)
        if row_at is None:
            row_at = 0 if middle_y < rows[-1] else len(rows) - 2
        if column_at is None:
            column_at = 0 if middle_x < columns[0] else len(columns) - 2
        cells[len(rows) - 2 - row_at][column_at].append(text)
    return [[" ".join(cell) for cell in row] for row in cells]


def _words_of(textpage) -> list[tuple]:
    """The words of a page with the box each one occupies.

    Built from the characters because PDFium reports boxes per character and
    per line, and a word is what must not be cut in half. A run of characters
    ends at whitespace.
    """
    out: list[tuple] = []
    try:
        total = textpage.count_chars()
    except Exception:  # noqa: BLE001
        return out

    letters: list[str] = []
    left = bottom = right = top = 0.0
    for index in range(total):
        try:
            character = textpage.get_text_range(index, 1)
            box = textpage.get_charbox(index, loose=False)
        except Exception:  # noqa: BLE001
            character, box = " ", None
        if character.strip() and box:
            if not letters:
                left, bottom, right, top = box
            else:
                left = min(left, box[0])
                bottom = min(bottom, box[1])
                right = max(right, box[2])
                top = max(top, box[3])
            letters.append(character)
        elif letters:
            out.append(("".join(letters), left, bottom, right, top))
            letters = []
    if letters:
        out.append(("".join(letters), left, bottom, right, top))
    return out


# How many of a grid's rows may hold more than one line of text before the
# thing is read as prose. A cell running to two lines is ordinary; most of them
# doing so means the horizontal rules were not row separators at all.
MULTILINE_ROWS_ALLOWED = 0.4


def _rows_are_single_lines(textpage, rows: list[float]) -> bool:
    """Whether the rules separate rows of a table rather than blocks of prose.

    A page can be ruled for reasons that have nothing to do with a table: a box
    around a worked example, a header rule, a footer rule. The text between two
    such rules is paragraphs, and cutting it into cells produces a well-formed
    table of shredded sentences -- "tons in the nucleus of an atom is its
    atomic number (Z)" arriving as four columns.

    A row of a table is a line. A block of prose is many. Counting the lines
    that fall between each pair of rules separates the two without needing to
    understand either.
    """
    boxes = _line_boxes(textpage)
    if not boxes:
        return True             # nothing to judge by; the other checks decide
    counted = crowded = 0
    for i in range(len(rows) - 1):
        top, bottom = rows[i], rows[i + 1]
        inside = [b[1] for b in boxes if bottom <= (b[1] + b[3]) / 2 <= top]
        if not inside:
            continue
        counted += 1
        # The cells of one row are reported as one box each, sitting on bases a
        # fraction of a point apart. Counting those as separate lines would call
        # every table prose, so they are grouped the way rules are.
        if len(_cluster(inside)) > 1:
            crowded += 1
    if not counted:
        return True
    return crowded <= counted * MULTILINE_ROWS_ALLOWED


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

    if not _rows_are_single_lines(textpage, rows):
        return None

    cells = _grid_cells(textpage, rows, columns)
    filled = sum(1 for row in cells for value in row if value)

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


# How many horizontal rules make a page look like it holds a table. Below this
# a page may be ruled for other reasons -- a header, a footer, a box around a
# note; at three or more the page is laid out the way a table is laid out.
RULES_SUGGEST_A_TABLE = 3


def examine(page, textpage=None) -> dict:
    """What this page has to say about its own tables.

    Returned together because finding out costs one walk over the objects of
    the page, which is the expensive part of reading it, and the caller needs
    all three: the table if there is one, whether the author labelled it, and
    whether the page is ruled the way a table is ruled.

    That last one matters because a caption count is not a census of tables.
    A book that labels a screenshot of a spreadsheet "Figure 4.2" announces no
    tables at all, and reading a count of zero as "there are none" leaves the
    whole document without the model ever looking at it. The rules are drawn
    whatever the author chose to call the thing.
    """
    if textpage is None:
        textpage = page.get_textpage()
    horizontals, verticals = _rules(page)
    rows, columns = _cluster(horizontals), _cluster(verticals)
    return {
        "table": _find_table(page, textpage, rows, columns),
        "announced": announces_a_table(textpage),
        "ruled": len(rows) >= RULES_SUGGEST_A_TABLE,
    }


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
    return _find_table(page, textpage, rows, columns)


def _find_table(page, textpage, rows: list[float],
                columns: list[float]) -> str | None:
    """The cascade itself, once the rules of the page have been read."""

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
