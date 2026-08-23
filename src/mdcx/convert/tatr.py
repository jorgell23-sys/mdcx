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

"""Reading the shape of a table that the drawn rules do not give away.

Most tables in academic material are found from what is drawn on the page --
see `tables`. What is left over is the borderless kind: a screenshot of a
spreadsheet, a layout held together by alignment alone. Guessing at those from
text alignment does not work, and the module next door explains at length why
not, so they are the one case that earns a model.

The model used here reads only the *shape*: where the rows are and where the
columns are. The words still come from the text layer of the PDF, as everywhere
else in this package, so nothing is transcribed and nothing can be hallucinated
into a cell. That also means a page with no text layer gains nothing here and
belongs to OCR instead.

It is the cheapest thing that does this job. Twenty-nine million parameters
against the hundreds of millions of a document-layout pipeline, and it runs on
the few pages that reached it rather than on all of them. Licences were checked
because this ships inside a library: the code is MIT, the weights are MIT, and
PubTables-1M, which trained it, is CDLA-Permissive-2.0.
"""
from __future__ import annotations

import threading

# Two models, because they answer different questions. The first is trained on
# whole pages and says where a table is; the second is trained on images that
# are already a table and says where its rows and columns run. Handing a whole
# page to the second one gets an answer -- it always gets an answer -- and the
# answer describes the page as though the paragraphs were columns.
DETECTOR = "microsoft/table-transformer-detection"
MODEL = "microsoft/table-transformer-structure-recognition"

# How much room to leave around a detected table when cutting it out. The
# structure model was trained on crops with a margin, and cutting flush loses
# the outermost row.
CROP_MARGIN = 12

# What the page is rasterised at. The processor resizes whatever it is given to
# an 800-pixel edge before the model sees it, so rendering larger than that is
# work thrown away twice -- once to draw the pixels and once to discard them.
# Measured over the pages of a textbook that reach this point, 150 dpi spent
# 154 ms a page and found five tables; 110 dpi spent 101 ms and found six.
DPI = 110

# Below this the detection is a guess, and a guess about the shape of a table
# produces a well-formed table of the wrong thing.
THRESHOLD = 0.7

# How many pages go to the accelerator at once.
#
# The cost of a page falls sharply with the size of the batch: measured, 150 ms
# sending one, 82 sending two, 75 with eight and 46 with twenty-four. What is
# being amortised is the fixed cost of the invocation, which is paid whole
# however few pages are in it.
#
# So eight was a floor written for a 6 GB card, not a property of the work, and
# a card with room to spare was being asked for eight pages at a time because
# that is what fitted somewhere else. What each page in flight costs is roughly
# what the models already hold divided by the batch they were measured with;
# BATCH_MIB is that, rounded up, and the batch is whatever the free memory
# allows between the floor and the ceiling.
#
# The ceiling exists because the curve flattens: past twenty-four the saving is
# small and the memory is not, and a batch large enough to fail is worse than a
# batch that leaves some of the card unused. The floor exists because a machine
# that reports very little free memory is usually one where something else is
# running, and a batch of one is the worst place on the curve.
BATCH_FLOOR = 8
BATCH_CEILING = 24

# What the models hold before any page is sent, and what each page in flight
# adds. Measured on a 6 GB card: 206 MiB with the models alone, 1,384 once the
# batch of eight is complete, which is 147 MiB a page.
MODELS_MIB = 206
BATCH_MIB = 150

# How much of the memory left over after seating the workers goes to larger
# batches. Half, because what a worker holds is an estimate and being wrong
# about it costs the whole run rather than one page.
SPARE_SHARE = 0.5


def _batch_size() -> int:
    """How many pages to send at once.

    The floor, unless someone who can see the whole run says otherwise.

    A worker cannot size this by itself, and trying was a mistake worth writing
    down. Reading the free memory and taking a share of it looks careful and is
    not: what is free is free for every process that may reach the model, not
    for this one, and the number of those is decided elsewhere. Sized that way
    on a 6 GB card, three workers each took a batch of seventeen -- some 2,756
    MiB apiece against a budget of 1,384 -- and the card went to 96% at 100%
    utilisation, which is the state this batching was meant to avoid.

    So the decision belongs where both halves of it are known: cli.py seats the
    workers on the card, then divides what is left over among them, and hands
    each the answer. MDCX_TATR_BATCH is how it is handed over, and also how a
    machine whose models measure differently says so.
    """
    import os

    declared = os.environ.get("MDCX_TATR_BATCH")
    if declared:
        try:
            return max(1, int(declared))
        except ValueError:
            pass
    return BATCH_FLOOR


def batch_for(free_mib: int | None, processes: int) -> int:
    """The batch that fits once `procesos` workers are seated on the card.

    Each worker is first guaranteed the floor, which is what the per-worker
    budget was measured with. Only what is left after seating all of them buys
    larger batches, and only half of that, because the measurement of what a
    worker holds is an estimate and the cost of being wrong is the whole run
    slowing down rather than one page.
    """
    if not free_mib or processes < 1:
        return BATCH_FLOOR
    settled = processes * (MODELS_MIB + BATCH_FLOOR * BATCH_MIB)
    leftover = free_mib - settled
    if leftover <= 0:
        return BATCH_FLOOR
    extra = int(leftover * SPARE_SHARE / processes / BATCH_MIB)
    return max(BATCH_FLOOR, min(BATCH_CEILING, BATCH_FLOOR + extra))


BATCH = _batch_size()

_LOADED: dict = {}
_LOCK = threading.RLock()


class MissingDependency(RuntimeError):
    """Raised when reading table shapes is asked for and cannot run."""


def available() -> bool:
    """Whether table shapes can be read in this interpreter."""
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load():
    """The processor and model, loaded once per process.

    Half precision on an accelerator and single precision on a processor, for
    the same reason as everywhere else: there is hardware for it in one place
    and not in the other.
    """
    with _LOCK:
        if MODEL in _LOADED:
            return _LOADED[MODEL]
        try:
            import torch
            from transformers import (AutoImageProcessor,
                                      TableTransformerForObjectDetection)
        except ImportError as e:
            raise MissingDependency(
                "Reading table shapes needs the tables extra: "
                "pip install 'mdcx[tables]'") from e

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def bring(name):
            processor = AutoImageProcessor.from_pretrained(name)
            model = TableTransformerForObjectDetection.from_pretrained(name)
            model = model.to(device).eval()
            if device == "cuda":
                model = model.half()
            return processor, model

        _LOADED[MODEL] = (bring(DETECTOR), bring(MODEL), device)
        return _LOADED[MODEL]


def _bounds(detected, labels: dict, wanted: str) -> list[tuple]:
    """The boxes the model gave for one kind of part."""
    out = []
    for label, box in zip(detected["labels"], detected["boxes"]):
        if labels[int(label)] == wanted:
            out.append(tuple(float(v) for v in box))
    return out


def _to_pdf(value: float, scale: float, height: float | None = None) -> float:
    """One image coordinate in the coordinates of the page.

    An image counts down from the top left; a PDF counts up from the bottom
    left. Passing the page height flips the axis; leaving it out is the
    horizontal case, where only the scale differs.
    """
    if height is None:
        return value / scale
    return height - value / scale




def _grid_from(detected, labels: dict, page, origin: tuple[float, float]
               ) -> tuple[list[float], list[float]] | None:
    """The rows and columns of a crop, in the coordinates of the page.

    The structure model works on an image of a table, so its coordinates are
    relative to where that image was cut from; `origin` puts them back on the
    page. They come back as boundaries rather than boxes, which is what cutting
    cells out of the text layer needs -- the grid described the same way a
    drawn one would be.
    """
    from .tables import _cluster

    rows = _bounds(detected, labels, "table row")
    columns = _bounds(detected, labels, "table column")
    if len(rows) < 2 or len(columns) < 2:
        return None

    scale = DPI / 72.0
    height = page.get_height()
    left_at, top_at = origin
    horizontals = _cluster(sorted(
        {_to_pdf(top_at + v, scale, height) for b in rows for v in (b[1], b[3])}))
    verticals = _cluster(sorted(
        {_to_pdf(left_at + v, scale) for b in columns for v in (b[0], b[2])}))
    if len(horizontals) < 3 or len(verticals) < 3:
        return None
    return sorted(horizontals, reverse=True), sorted(verticals)


def _cuts_fall_between_words(textpage, cells, rows, columns) -> bool:
    """Whether the column boundaries land between words rather than inside them.

    Reading each character into exactly one cell means nothing is duplicated,
    so a row always adds up letter for letter -- which makes that check useless
    here. What a shape read off a rendering gets wrong is different: a boundary
    a few points to one side falls inside a word and the halves end up in
    neighbouring cells. "quartile of" split that way rejoins as "quartileof",
    so the spaces are what tells the two apart.

    The row is read again as a single cell spanning the whole width, by the
    same rule, and the two are compared: a genuine boundary sits where there
    was already a space.
    """
    from .tables import _grid_cells

    for i, row in enumerate(cells):
        if not any(row):
            continue
        whole = _grid_cells(textpage, [rows[i], rows[i + 1]],
                            [columns[0], columns[-1]])[0][0]
        if " ".join(v for v in row if v) != whole:
            return False
    return True


def _as_markdown(textpage, rows: list[float], columns: list[float]) -> str | None:
    """The table, or None where the shape does not survive the text under it."""
    from .tables import MIN_OCCUPANCY, cells_by_word

    # Whole words, not single characters. This grid was read off a rendering,
    # so its boundaries sit wherever the pixels put them -- often a point or
    # two inside a word, which placing characters one by one would split.
    cells = cells_by_word(textpage, rows, columns)
    filled = sum(1 for row in cells for value in row if value)
    wide = (len(rows) - 1) * (len(columns) - 1)
    if not wide or filled < wide * MIN_OCCUPANCY:
        # The shape is there and the words are not: a page with no text layer,
        # which is a job for OCR and not for this.
        return None
    body = ["| " + " | ".join(row) + " |" for row in cells if any(row)]
    if len(body) < 2:
        return None
    body.insert(1, "|" + "---|" * (len(columns) - 1))
    return "\n".join(body)


def _band_from_rules(page, image) -> tuple[float, float, float, float] | None:
    """Where the drawn rules put the table, in the coordinates of the image.

    A page ruled like a table has already answered the question the detector
    is asked, and answered it exactly rather than approximately. Reusing that
    saves a pass over the page for every page that draws anything -- which,
    among the pages that get this far, is most of them.
    """
    from .tables import RULES_SUGGEST_A_TABLE, _cluster, _rules

    horizontals, _ = _rules(page)
    rows = _cluster(horizontals)
    if len(rows) < RULES_SUGGEST_A_TABLE:
        return None
    scale = DPI / 72.0
    height = page.get_height()
    top = (height - max(rows)) * scale
    bottom = (height - min(rows)) * scale
    if bottom - top < 20:
        return None
    return 0.0, top, float(image.width), bottom


def tables_on_pages(pages: list, textpages: list) -> list:
    """The table on each page, or None where no table survives the reading.

    Two passes. The first looks at whole pages and says where a table is; the
    second looks at each table on its own and says how it is ruled. They are
    separate models because they were trained on different things, and giving
    the second one a whole page yields a confident description of the
    paragraphs as columns.

    Batched, because the accelerator sits idle between one page and the next
    otherwise, and these are pages already known to be worth the trouble.
    """
    import torch

    (find_proc, finder), (read_proc, reader), device = load()
    out: list = [None] * len(pages)

    def look(processor, model, images):
        with torch.no_grad():
            prepared = processor(images=images, return_tensors="pt").to(device)
            if device == "cuda":
                prepared["pixel_values"] = prepared["pixel_values"].half()
            return processor.post_process_object_detection(
                model(**prepared), threshold=THRESHOLD,
                target_sizes=[(im.height, im.width) for im in images])

    for start in range(0, len(pages), BATCH):
        group = pages[start:start + BATCH]
        images = [p.render(scale=DPI / 72.0).to_pil() for p in group]

        # Where the page draws rules, they already say where the table is, and
        # asking a model the same question costs as much as reading the shape
        # afterwards. The detector is for the pages that draw nothing.
        drawn = [_band_from_rules(page, image) for page, image in zip(group, images)]
        ask = [i for i, band in enumerate(drawn) if band is None]
        found: list = [None] * len(group)
        if ask:
            for detected, i in zip(look(find_proc, finder,
                                        [images[i] for i in ask]), ask):
                found[i] = detected

        crops, belongs_to = [], []
        for offset in range(len(group)):
            frame = drawn[offset]
            if frame is None:
                detected = found[offset]
                frames = _bounds(detected, finder.config.id2label, "table") if detected else []
                if not frames:
                    continue
                frame = max(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            image = images[offset]
            left = max(0, int(frame[0]) - CROP_MARGIN)
            top = max(0, int(frame[1]) - CROP_MARGIN)
            right = min(image.width, int(frame[2]) + CROP_MARGIN)
            bottom = min(image.height, int(frame[3]) + CROP_MARGIN)
            if right - left < 40 or bottom - top < 40:
                continue
            crops.append(image.crop((left, top, right, bottom)))
            belongs_to.append((start + offset, (left, top)))

        for at in range(0, len(crops), BATCH):
            batch = crops[at:at + BATCH]
            read = look(read_proc, reader, batch)
            for offset, detected in enumerate(read):
                index, origin = belongs_to[at + offset]
                shape = _grid_from(detected, reader.config.id2label,
                                   pages[index], origin)
                if not shape:
                    continue
                out[index] = _as_markdown(textpages[index], *shape)
    return out
