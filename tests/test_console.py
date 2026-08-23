"""Checks that reporting the work cannot stop the work.

A console has a code page, and on Windows it is usually one that cannot
represent every character a document name may contain. Printing the name of a
file called `Fizyka dla szkół wyższych — łódź.pdf` then raises, and what fails
is showing the conversion rather than performing it.

Of 195 files taken from open-access catalogues, 24 per cent had names outside
ASCII. The case is ordinary, not exotic.

Two failures of this kind were found. In the parent process it ended the run. In
the worker it was worse: the line was printed before the conversion was
attempted and outside the guard, so a name the console could not represent meant
the document was never converted, and was recorded as an engine failure.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import console  # noqa: E402
from mdcx.convert.convert import convert_one_safe  # noqa: E402
from mdcx.convert.paths import plan_jobs  # noqa: E402

# A title from the OpenStax catalogue in Polish. Every character outside cp1252
# here is one that appears in real catalogue entries.
POLISH_NAME = "Fizyka dla szkół wyższych — łódź"


class LimitedConsole(io.TextIOBase):
    """A stream with a code page that cannot represent every character.

    It has no reconfigure, which is the case that the fallback exists for: a
    stream replaced by a wrapper or a harness cannot be switched to UTF-8.
    """

    encoding = "cp1252"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        text.encode("cp1252")  # raises exactly as the console would
        self.written.append(text)
        return len(text)


def test_a_limited_console_would_raise():
    """The premise of these tests: this stream fails as the console fails."""
    stream = LimitedConsole()
    with pytest.raises(UnicodeEncodeError):
        stream.write(POLISH_NAME)


def test_safe_print_degrades_instead_of_raising(monkeypatch):
    """The unrepresentable character is replaced; the line still appears."""
    stream = LimitedConsole()
    monkeypatch.setattr(sys, "stdout", stream)
    console.safe_print(f">>> {POLISH_NAME}")
    written = "".join(stream.written)
    assert ">>>" in written
    assert "Fizyka" in written
    assert "wy" in written, "the representable part of the name survives"


def test_safe_print_passes_its_arguments(monkeypatch):
    stream = LimitedConsole()
    monkeypatch.setattr(sys, "stdout", stream)
    console.safe_print("plain text", end="")
    assert "".join(stream.written) == "plain text"


def test_configure_survives_a_stream_that_cannot_be_reconfigured(monkeypatch):
    """Configuration must not raise on a stream that does not support it."""
    monkeypatch.setattr(sys, "stdout", LimitedConsole())
    monkeypatch.setattr(sys, "stderr", LimitedConsole())
    console.configure()


def test_a_name_outside_the_code_page_still_converts(monkeypatch):
    """The document is converted, whatever the console can display.

    This is the failure that mattered: the worker printed the name before
    attempting the conversion and outside the guard, so the document was
    recorded as an engine failure without the conversion ever being tried.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entry = root / "in"
        entry.mkdir()
        (entry / f"{POLISH_NAME}.txt").write_text(
            "Content long enough for the conversion to have something to do.",
            encoding="utf-8")
        output = root / "out"
        output.mkdir()

        jobs, _ = plan_jobs(entry)
        assert jobs, "the file must be queued"

        monkeypatch.setattr(sys, "stdout", LimitedConsole())
        record = convert_one_safe((jobs[0], output, False, False, True))

        assert record["engine"] != "none", (
            f"the conversion was not attempted: {record.get('errors')}")
        assert record["ok"], record.get("errors")
        assert (output / f"{POLISH_NAME}.md").exists()


def test_the_written_document_keeps_its_name(monkeypatch):
    """The output file is named after the source, not after what fits."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entry = root / "in"
        entry.mkdir()
        (entry / f"{POLISH_NAME}.txt").write_text("Some content here.",
                                                      encoding="utf-8")
        output = root / "out"
        output.mkdir()
        jobs, _ = plan_jobs(entry)
        monkeypatch.setattr(sys, "stdout", LimitedConsole())
        convert_one_safe((jobs[0], output, False, False, True))
        names = [p.name for p in output.glob("*.md")]
        assert f"{POLISH_NAME}.md" in names
