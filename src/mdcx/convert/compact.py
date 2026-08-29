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

"""Removal of Markdown scaffolding that is not content.

Conversion engines pad table cells so that columns align in the file, emit
separator rows with dozens of dashes per column, keep the full grid even when
entire columns hold no data, and insert their own markers where they could
not transcribe. None of that comes from the source document.

Compaction is applied per file and verified in the same operation: if the
result were to lose any word, the original text is kept.
"""
from __future__ import annotations

import re

from . import verify

HEADER_FIELDS = (
    "source:",
    "source_pseudopath:",
    "source_format:",
    "pages:",
    # Kept for the same reason `pages` is, and useless without it: on their own,
    # twelve pages read as a document of twelve pages. These two are what say it
    # is twelve pages *of eighty*, which is the difference between a sample and
    # a short book -- and compaction is on by default, so dropping them here
    # would mean the distinction never survives to anyone reading the corpus.
    "pages_total:",
    "sampled:",
    "verification_status:",
    "chapter_title:",
)

def compact_tables(text: str) -> str:
    """Remove the padding that aligns table columns."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s[1:-1].split("|")]
            out.append("| " + " | ".join(cells) + " |")
        else:
            out.append(line)
    return "\n".join(out)

def compact_separators(text: str) -> str:
    """Reduce a table separator row to the minimum Markdown requires."""
    def _sep(m: re.Match) -> str:
        columns = m.group(0).strip().strip("|").split("|")
        return "|" + "|".join("---" for _ in columns) + "|"

    return re.sub(r"(?m)^[ \t]*\|[\s\-:|]+\|[ \t]*$", _sep, text)

def drop_markers(text: str) -> str:
    """Remove converter markers such as image placeholders."""
    return re.sub(r"(?m)^[ \t]*<!--\s*(image|imagen)\s*-->[ \t]*\n?", "", text)

def collapse_blanks(text: str) -> str:
    """Collapse consecutive blank lines and trailing spaces."""
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text)

def compress_header(text: str) -> str:
    """Reduce the metadata header to the fields needed to cite and audit."""
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    cab, body = parts[1], parts[2]
    lines = [l for l in cab.splitlines() if l.strip().startswith(HEADER_FIELDS)]
    if not lines:
        return text
    return "---\n" + "\n".join(lines) + "\n---" + body

def drop_empty_columns(text: str) -> str:
    """Remove table columns that contain no data."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("|"):
            j += 1
        table = lines[i:j]
        rows = [[c.strip() for c in l.strip()[1:-1].split("|")] for l in table]
        width = max((len(f) for f in rows), default=0)
        is_sep = [bool(re.fullmatch(r"[\s\-:]*", "".join(f))) for f in rows]
        useful = [any(len(f) > c and f[c] and not is_sep[k] for k, f in enumerate(rows))
                for c in range(width)]
        if not any(useful) or all(useful):
            out.extend(table)
        else:
            for k, f in enumerate(rows):
                cells = [f[c] if len(f) > c else "" for c in range(width) if useful[c]]
                if is_sep[k]:
                    out.append("|" + "|".join("---" for _ in cells) + "|")
                else:
                    out.append("| " + " | ".join(cells) + " |")
        i = j
    return "\n".join(out)

STEPS = (
    compact_tables,
    compact_separators,
    drop_markers,
    collapse_blanks,
    compress_header,
    drop_empty_columns,
)

def compact(text: str) -> tuple[str, bool]:
    """Compact the Markdown and verify in the same operation that no content was lost."""
    try:
        compressed = text
        for step in STEPS:
            compressed = step(compressed)
        compressed = compressed.rstrip("\n") + "\n"
    except Exception:  # noqa: BLE001
        return text, False

    before = verify.tokenize(verify.strip_markdown_noise(_body(text)))
    after = verify.tokenize(verify.strip_markdown_noise(_body(compressed)))
    if _is_missing(before, after):
        return text, False
    return compressed, True

def _body(text: str) -> str:
    """Return the text without the metadata header."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2]
    return text

def _is_missing(before: list[str], after: list[str]) -> bool:
    """Report whether any word of the original text is missing, counting repetitions."""
    from collections import Counter

    missing = Counter(before)
    missing.subtract(Counter(after))
    return any(n > 0 for n in missing.values())
