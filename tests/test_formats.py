"""Checks that a file is treated as what it contains, not as what it is called.

Deciding by extension has two failure modes and both were observed in the wild.
A supported format under an unexpected extension is discarded although the
engine can read it, and a file served under the wrong extension is handed to an
engine that cannot open it, which fails in a way that looks like a damaged
document rather than a misrouted one.

The second is not hypothetical: open-access repositories serve EPUB files from
URLs ending in .pdf, declaring application/pdf, where only the bytes disagree.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import extract  # noqa: E402
from mdcx.convert.paths import SUPPORTED, plan_jobs, resolve_format, sniff_format  # noqa: E402


def build_epub(path: Path, title: str = "A Book",
               body: str = "The quick brown fox jumps over the lazy dog.") -> Path:
    """Write a minimal but valid EPUB."""
    with zipfile.ZipFile(path, "w") as archive:
        # The specification requires this member first and stored, not deflated.
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml",
                         '<?xml version="1.0"?><container version="1.0">'
                         '<rootfiles><rootfile full-path="OEBPS/content.opf"/>'
                         "</rootfiles></container>")
        archive.writestr("OEBPS/content.opf",
                         '<?xml version="1.0"?><package version="3.0">'
                         f"<metadata><title>{title}</title></metadata>"
                         '<manifest><item id="c1" href="chapter1.xhtml"/></manifest>'
                         '<spine><itemref idref="c1"/></spine></package>')
        archive.writestr("OEBPS/chapter1.xhtml",
                         "<html><head><title>ignored</title>"
                         "<style>body { color: red }</style></head>"
                         f"<body><h1>{title}</h1><p>{body}</p></body></html>")
    return path


def build_docx_like(path: Path) -> Path:
    """A ZIP shaped like an OOXML document, enough to be identified as one."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    return path


def test_epub_is_a_supported_format():
    assert SUPPORTED[".epub"] == "epub"


def test_content_identifies_an_epub(tmp_path):
    book = build_epub(tmp_path / "book.epub")
    assert sniff_format(book) == "epub"


def test_an_epub_named_pdf_is_still_an_epub(tmp_path):
    """The bytes decide, not the name.

    Repositories serve EPUB files from .pdf URLs. Trusting the name sends the
    file to the PDF reader, which cannot open it.
    """
    disguised = build_epub(tmp_path / "served_as.pdf")
    assert sniff_format(disguised) == "epub"
    assert resolve_format(disguised) == "epub"


def test_ooxml_is_identified_by_its_members(tmp_path):
    document = build_docx_like(tmp_path / "report.docx")
    assert sniff_format(document) == "docx"


def test_a_pdf_is_identified_by_its_header(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    assert sniff_format(pdf) == "pdf"


def test_text_falls_back_to_the_extension(tmp_path):
    """Plain text carries no signature, so its name is the only information."""
    note = tmp_path / "note.txt"
    note.write_text("plain text carries no signature", encoding="utf-8")
    assert sniff_format(note) is None
    assert resolve_format(note) == "text"


def test_an_unknown_file_is_not_guessed(tmp_path):
    unknown = tmp_path / "data.bin"
    unknown.write_bytes(b"\x00\x01\x02\x03binary")
    assert resolve_format(unknown) is None


def test_a_damaged_zip_is_not_claimed(tmp_path):
    """A truncated archive identifies nothing and must not be forced through."""
    damaged = tmp_path / "broken.epub"
    damaged.write_bytes(b"PK\x03\x04" + b"\x00" * 40)
    assert sniff_format(damaged) is None


def test_planning_queues_epub_under_either_name(tmp_path):
    """Both the honest and the misnamed EPUB reach the queue, as EPUB."""
    entry = tmp_path / "input"
    entry.mkdir()
    build_epub(entry / "honest.epub")
    build_epub(entry / "misnamed.pdf")
    (entry / "note.txt").write_text("text", encoding="utf-8")

    jobs, skipped = plan_jobs(entry)
    formats = {j.source.name: j.kind for j in jobs}
    assert not skipped
    assert formats["honest.epub"] == "epub"
    assert formats["misnamed.pdf"] == "epub"
    assert formats["note.txt"] == "text"


def test_reference_text_of_an_epub_is_measurable(tmp_path):
    """An EPUB yields reference text, so its conversion can be verified.

    Without it the coverage of an EPUB is undefined, and a document whose
    fidelity cannot be measured is one nobody can audit afterwards.
    """
    book = build_epub(tmp_path / "book.epub", title="Measured",
                       body="Photosynthesis lets plants turn sunlight into sugar.")
    text, meta = extract.reference_text(book, "epub")
    assert "error" not in meta
    assert meta["documents"] == 1
    assert "Photosynthesis" in text
    assert "Measured" in text


def test_reference_text_leaves_out_markup_and_code(tmp_path):
    """Style and markup are not content, and counting them would credit noise."""
    book = build_epub(tmp_path / "book.epub")
    text, _ = extract.reference_text(book, "epub")
    assert "color: red" not in text
    assert "<p>" not in text
    assert "ignored" not in text, "the head title is not body text"


def test_a_damaged_epub_reports_instead_of_raising(tmp_path):
    """Extraction never raises: a failure is reported as a failure."""
    damaged = tmp_path / "broken.epub"
    damaged.write_bytes(b"PK\x03\x04 not really an archive")
    text, meta = extract.reference_text(damaged, "epub")
    assert text == ""
    assert "error" in meta


def test_an_epub_missing_its_package_file_still_reads(tmp_path):
    """The reading order is a hint. Without it the documents are still content."""
    path = tmp_path / "no_opf.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("text/one.xhtml", "<html><body><p>First part.</p></body></html>")
        archive.writestr("text/two.xhtml", "<html><body><p>Second part.</p></body></html>")
    text, meta = extract.reference_text(path, "epub")
    assert meta["documents"] == 2
    assert "First part." in text and "Second part." in text
