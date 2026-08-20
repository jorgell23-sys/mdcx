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

"""Fidelity verification of a conversion.

Coverage is measured as a multiset of words: how much of the reference text
appears in the Markdown, counting repetitions. Numeric tokens are reported
separately, since a lost figure matters more than a lost article.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

COVERAGE_OK = 0.995
COVERAGE_WARN = 0.970

# The tokenizer of the retrieval engine, imported rather than restated, so that a
# document verified as complete is a document that can be searched. Restating it
# splits words wherever the two patterns disagree, and the count of a word that
# was never a word measures nothing.
from ..search import _TOKEN_RE  # noqa: E402

_MD_NOISE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\|[-: |]+\|"              # a table separator row: |---|---|
    r"|`{3,}[A-Za-z0-9_+.-]*"   # a line that is only a code fence
    r"|<!--[^\n]*?-->"          # a converter marker, such as <!-- image -->
    r")[ \t]*$"
)

def tokenize(text: str) -> Counter:
    text = unicodedata.normalize("NFC", text)
    return Counter(m.group(0).lower() for m in _TOKEN_RE.finditer(text))

def strip_markdown_noise(md: str) -> str:
    """Remove Markdown scaffolding before comparing content."""
    return _MD_NOISE.sub(" ", md)

_strip_markdown_noise = strip_markdown_noise

def compare(reference: str, markdown: str, sample: int = 25) -> dict:
    """Compare Markdown against the reference text and return coverage."""
    ref = tokenize(reference)
    got = tokenize(strip_markdown_noise(markdown))

    total = sum(ref.values())
    if total == 0:
        produced = sum(got.values())
        return {
            "measurable": False,
            "status": "no-reference",
            "coverage": None,
            "numeric_coverage": None,
            "ref_tokens": 0,
            "md_tokens": produced,
            "missing_sample": [],
            "note": "the original exposes no text: fidelity not verifiable by tokens",
        }

    matched = sum(min(count, got[token]) for token, count in ref.items())
    coverage = matched / total

    num_ref = {t: c for t, c in ref.items() if any(ch.isdigit() for ch in t)}
    num_total = sum(num_ref.values())
    num_matched = sum(min(c, got[t]) for t, c in num_ref.items())
    numeric_coverage = (num_matched / num_total) if num_total else None

    deficits = [
        (count - got[token], token)
        for token, count in ref.items()
        if got[token] < count
    ]
    deficits.sort(reverse=True)

    if coverage >= COVERAGE_OK:
        status = "ok"
    elif coverage >= COVERAGE_WARN:
        status = "warn"
    else:
        status = "fail"

    if numeric_coverage is not None and numeric_coverage < COVERAGE_WARN and status == "ok":
        status = "warn"

    return {
        "measurable": True,
        "status": status,
        "coverage": round(coverage, 5),
        "numeric_coverage": round(numeric_coverage, 5) if numeric_coverage is not None else None,
        "ref_tokens": total,
        "md_tokens": sum(got.values()),
        "missing_tokens": sum(d for d, _ in deficits),
        "missing_sample": [t for _, t in deficits[:sample]],
    }
