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

"""Rasterise the alphabet and group the letters by shape, beside the frozen table.

`mdcx.shapekey.TABLE` maps each character onto the class of its printed shape --
how many groups of dots touch the top of the glyph, and how many the bottom. It
ships frozen, because deriving it needs a font file, an imaging library and a
rasteriser, none of which belong in a package that has to install anywhere.

**This does not reproduce that table, and does not try to.** Run against Times it
groups most of the alphabet the same way and moves a handful of characters, and
that is the expected result rather than a fault: a signature depends on the
typeface, on where the grid is cropped, on the threshold that calls a pixel a
dot, and on which of two readings of "the top" is taken -- the top of the glyph
or the top of the line. The essay this came from measured that ambiguity:
between Times and Georgia only 55 per cent of the signatures agree.

So this shows the derivation; it does not certify the table. What the table has
to be right about is narrower, and is checked where it belongs, in
`tests/test_shapekey.py`: that the characters optical recognition confuses land
in one bucket -- 0 with o, 1 with l, 5 with s. That property is what the sieve
rests on, and it does not depend on reproducing anyone's rasteriser.

    python tools/derive_shape_table.py                 # group with a typeface
    python tools/derive_shape_table.py --print         # emit the literal

It needs Pillow and a font, and says so plainly when it has neither rather than
reporting agreement it did not measure.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import shapekey  # noqa: E402

# The order of the matrices dot printers used, which is where the idea comes
# from: nine rows of seven dots.
ROWS, COLUMNS = 9, 7

# Times first, because that is what the shipped table was derived from. The
# others are here to show how much the answer moves between typefaces, which is
# the finding that bounds the method rather than a defect in it.
FONTS = ("times.ttf", "cour.ttf", "georgia.ttf", "cambria.ttc")


def _font_paths(named: str | None) -> list[Path]:
    if named:
        return [Path(named)]
    found = []
    for name in FONTS:
        for path in glob.glob(str(Path("C:/Windows/Fonts") / name)):
            found.append(Path(path))
    return found


def _grid(character: str, font) -> list[list[bool]]:
    """The character rendered onto a ROWS x COLUMNS grid of on and off.

    Rendered large and then reduced, rather than rendered small: a glyph drawn
    directly at seven pixels wide loses the strokes this counts.
    """
    from PIL import Image, ImageDraw

    size = 128
    image = Image.new("L", (size, size), 255)
    ImageDraw.Draw(image).text((size // 4, size // 8), character,
                               font=font, fill=0)
    box = image.point(lambda v: 255 if v < 128 else 0).getbbox()
    if box is None:            # a character this font does not draw
        return []
    cropped = image.crop(box).resize((COLUMNS, ROWS))
    pixels = cropped.load()
    return [[pixels[x, y] < 160 for x in range(COLUMNS)] for y in range(ROWS)]


def _groups(row: list[bool]) -> int:
    """Runs of adjacent dots in one row: two strokes with a gap count as two."""
    count, inside = 0, False
    for on in row:
        if on and not inside:
            count += 1
        inside = on
    return count


def derive(font_path: Path) -> dict:
    """The class of every character in the table, read off this typeface."""
    from PIL import ImageFont

    font = ImageFont.truetype(str(font_path), 96)
    signatures: dict[str, tuple[int, int]] = {}
    for character in sorted(shapekey.TABLE):
        grid = _grid(character, font)
        if not grid:
            continue
        signatures[character] = (_groups(grid[0]), _groups(grid[-1]))

    # The classes are named in the order they are first met, which is what makes
    # two derivations comparable at all: the letters are what matter, not which
    # letter of the alphabet each class happens to be called.
    names: dict[tuple[int, int], str] = {}
    table: dict[str, str] = {}
    for character, signature in signatures.items():
        if signature not in names:
            names[signature] = "abcdefghijkl"[len(names)]
        table[character] = names[signature]
    return table


def _same_partition(one: dict, other: dict) -> bool:
    """Whether two tables group the characters the same way.

    Compared as partitions and not as literals, because the class names are
    arbitrary: what the table asserts is which characters share a shape, and
    renaming the classes asserts nothing different.
    """
    def blocks(table):
        out: dict[str, set] = {}
        for character, cls in table.items():
            out.setdefault(cls, set()).add(character)
        return sorted(map(sorted, out.values()))

    return blocks(one) == blocks(other)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", help="a .ttf to use instead of the defaults")
    parser.add_argument("--print", action="store_true", dest="emit",
                        help="print the derived table as a Python literal")
    args = parser.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow is not installed, so the table cannot be derived. "
              "It ships frozen and is used from mdcx.shapekey; this tool is "
              "how it is checked, not how it is read.", file=sys.stderr)
        return 2

    fonts = [p for p in _font_paths(args.font) if p.is_file()]
    if not fonts:
        print("No font found to derive from. Pass --font with a .ttf.",
              file=sys.stderr)
        return 2

    reference = None
    for path in fonts:
        derived = derive(path)
        if reference is None:
            reference = derived
            agrees = _same_partition(derived, shapekey.TABLE)
            print(f"{path.name}: groups the alphabet "
                  f"{'exactly as' if agrees else 'differently from'} "
                  f"the frozen table")
            if not agrees:
                shipped = {c: shapekey.TABLE[c] for c in sorted(derived)}
                for character in sorted(derived):
                    here = {k for k, v in derived.items() if v == derived[character]}
                    there = {k for k, v in shipped.items()
                             if v == shipped.get(character)}
                    if here != there:
                        print(f"  {character!r}: grouped with "
                              f"{''.join(sorted(here - {character}))} here, "
                              f"{''.join(sorted(there - {character}))} in the table")
            if args.emit:
                print("TABLE = {")
                for character in sorted(derived):
                    print(f'    {character!r}: {derived[character]!r},')
                print("}")
        else:
            # Compared by class-mates, not by class name. Asking whether two
            # tables give a character the same label answers nothing -- the
            # labels are assigned in encounter order and mean nothing across
            # runs. What a character has in common between two typefaces is the
            # set of characters drawn like it.
            def mates(table, character):
                return {k for k, v in table.items()
                        if v == table[character]} - {character}

            shared = sum(1 for c in derived if c in reference
                         and mates(derived, c) == mates(reference, c))
            common = sum(1 for c in derived if c in reference)
            print(f"{path.name}: groups {shared} of {common} characters with "
                  f"the same letters as {fonts[0].name}")

    print("\nA signature is a property of the typeface and not of the letter, "
          "which is why the table is frozen against one: the A of Times has one "
          "group above and two below, not two and two, because its apex is "
          "pointed. Differences above are expected and say nothing about "
          "whether the shipped table works. What it has to be right about is "
          "that the characters optical recognition confuses share a bucket, "
          "and that is checked in tests/test_shapekey.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
