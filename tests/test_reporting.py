"""Checks that the conversion summary separates what it means to separate.

A document that exposes no text was never measured against an original. That is
not the same as a document that was measured and fell short, and counting the
two together misreports a collection: a set of books with a scanned plate was
reported as entirely non-conforming while its coverage read 100 per cent,
because coverage is computed over the documents that could be measured at all.

At the scale these collections reach, the difference decides whether a run looks
like a success or a failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import index  # noqa: E402


def record(name: str, ok: bool, status: str, coverage: float | None = None,
           ref_tokens: int = 0, missing: int = 0) -> dict:
    """A conversion record as the converter writes it."""
    return {
        "source_name": name,
        "source_pseudopath": f"@/{name}",
        "markdown_pseudopath": f"@/{name}.md",
        "ok": ok,
        "engine": "docling",
        "verification": {
            "status": status,
            "measurable": coverage is not None,
            "coverage": coverage,
            "numeric_coverage": None,
            "ref_tokens": ref_tokens,
            "missing_tokens": missing,
        },
    }


def summarise(records: list[dict]) -> dict:
    manifest = index.build_manifest(records, Path("."), [], 1.0)
    return manifest["summary"]


def test_a_measured_document_that_meets_its_original_is_conforming():
    res = summarise([record("a.pdf", True, "ok", 1.0, 100, 0)])
    assert res["converted_ok"] == 1
    assert res["with_findings"] == 0
    assert res.get("unverifiable", 0) == 0


def test_a_measured_document_that_falls_short_has_findings():
    res = summarise([record("a.pdf", False, "warn", 0.82, 100, 18)])
    assert res["converted_ok"] == 0
    assert res["with_findings"] == 1
    assert res.get("unverifiable", 0) == 0


def test_a_document_with_no_text_is_unverifiable_rather_than_failing():
    """A scan has no original text to measure against.

    Reporting it as a finding says the conversion fell short, which is a claim
    about a comparison that was never made.
    """
    res = summarise([record("scan.pdf", False, "no-reference")])
    assert res["converted_ok"] == 0
    assert res["with_findings"] == 0
    assert res["unverifiable"] == 1


def test_an_unreadable_document_is_unverifiable_too():
    res = summarise([record("blank.pdf", False, "unreadable")])
    assert res["with_findings"] == 0
    assert res["unverifiable"] == 1


def test_the_three_counts_partition_the_collection():
    """Every document falls in exactly one of the three, and they sum."""
    res = summarise([
        record("good.pdf", True, "ok", 1.0, 100, 0),
        record("short.pdf", False, "warn", 0.7, 100, 30),
        record("scan.pdf", False, "no-reference"),
        record("blank.pdf", False, "unreadable"),
    ])
    assert res["documents"] == 4
    assert res["converted_ok"] == 1
    assert res["with_findings"] == 1
    assert res["unverifiable"] == 2
    assert (res["converted_ok"] + res["with_findings"]
            + res["unverifiable"]) == res["documents"]


def test_coverage_is_measured_over_what_could_be_measured():
    """An unverifiable document neither raises nor lowers coverage.

    It contributes no reference tokens, so including it would require deciding
    what fraction of nothing was recovered.
    """
    with_scan = summarise([
        record("good.pdf", True, "ok", 1.0, 100, 0),
        record("scan.pdf", False, "no-reference"),
    ])
    only_measurable = summarise([record("good.pdf", True, "ok", 1.0, 100, 0)])
    assert with_scan["global_token_coverage"] == only_measurable["global_token_coverage"]
