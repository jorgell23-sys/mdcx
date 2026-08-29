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

"""How the key gets in, and what it is allowed to be.

Two faults, found in one sitting and touching the same surface.

An empty key was accepted end to end. scrypt derives from b"" as happily as
from anything else, so the package was written, encrypted, reported as packed
-- and opened again by anyone who thought to try the empty string. It was found
by accident: a badly quoted pipeline left an environment variable empty and the
packaging reported success.

And the key arrived only through --key, which puts it on the command line, where
any process on the machine can read it. Packaging a large corpus takes tens of
minutes, and that is how long the secret sits in the process table. It was found
by running ps to check on progress.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nSome text.\n", encoding="utf-8")
    (folder / "two.md").write_text("# Two\n\nOther text.\n", encoding="utf-8")
    return folder


# --- An empty key is not a key ----------------------------------------------


@pytest.mark.parametrize("empty", ["", "   ", "\n", "\t"])
def test_packing_refuses_a_key_that_is_not_one(corpus, tmp_path, empty):
    """The package would be encrypted with no secret, and nothing would say so."""
    with pytest.raises(ValueError) as refused:
        archive.pack(corpus, tmp_path / "empty.mdcx", empty)

    assert "empty" in str(refused.value).lower(), (
        "the refusal has to say what happened; the case that produced this was "
        "a variable that resolved to nothing")
    assert not (tmp_path / "empty.mdcx").exists(), "a useless package was written"


def test_a_real_key_still_packs(corpus, tmp_path):
    """The refusal must not reach anything that was working."""
    target = tmp_path / "real.mdcx"
    archive.pack(corpus, target, "a real passphrase")

    connection, header = archive.open_package(target, "a real passphrase")
    assert header.get("documents") == 2


def test_a_package_already_written_that_way_still_opens(corpus, tmp_path):
    """Deliberately not symmetric, and fixed here so it is not "corrected".

    Refusing the empty string on open would make unreadable the packages
    already written with it, which is worse than being able to open them. What
    had to be stopped is creating them.

    Opening is checked here by what it says when the empty string is wrong: if
    open_package had grown a refusal of its own, it would complain about the
    key being empty instead of trying it and failing to decrypt. Trying it is
    what keeps the old packages readable.
    """
    target = tmp_path / "real.mdcx"
    archive.pack(corpus, target, "a real passphrase")

    with pytest.raises(ValueError) as refused:
        archive.open_package(target, "")

    said = str(refused.value).lower()
    assert "empty" not in said, (
        "opening now refuses the empty string, which makes unreadable every "
        "package that was written with it")
    assert "incorrect key" in said or "corrupted" in said


# --- The key does not have to travel in argv --------------------------------


def _args(**fields):
    return argparse.Namespace(**{"key": None, "key_file": None, **fields})


def test_the_environment_is_read(monkeypatch):
    """MDCX_KEY, which the MCP server already reads: the convention existed."""
    monkeypatch.setenv("MDCX_KEY", "from the environment")
    assert archive.resolve_key(_args()) == "from the environment"


def test_a_file_is_read(tmp_path, monkeypatch):
    """Where the key already lives in a file with permissions of its own."""
    monkeypatch.delenv("MDCX_KEY", raising=False)
    keyfile = tmp_path / "key.txt"
    keyfile.write_text("from a file\n", encoding="utf-8")

    assert archive.resolve_key(_args(key_file=str(keyfile))) == "from a file"


def test_standard_input_is_read(monkeypatch):
    """--key - is what tools that handle secrets offer."""
    monkeypatch.delenv("MDCX_KEY", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO("from standard input\n"))

    assert archive.resolve_key(_args(key="-")) == "from standard input"


def test_the_command_line_still_works(monkeypatch):
    """Removing --key would break everyone, and it is fine on one's own machine."""
    monkeypatch.setenv("MDCX_KEY", "ignored when --key is explicit")
    assert archive.resolve_key(_args(key="on the command line")) == (
        "on the command line")


def test_a_file_wins_over_the_command_line(tmp_path, monkeypatch):
    """The ways that keep the secret off the command line come first."""
    monkeypatch.delenv("MDCX_KEY", raising=False)
    keyfile = tmp_path / "key.txt"
    keyfile.write_text("from a file", encoding="utf-8")

    assert archive.resolve_key(
        _args(key="on the command line", key_file=str(keyfile))) == "from a file"


def test_no_key_at_all_says_the_ways_there_are(monkeypatch):
    """--key stopped being required, so the message has to teach the rest."""
    monkeypatch.delenv("MDCX_KEY", raising=False)

    with pytest.raises(SystemExit) as stopped:
        archive.resolve_key(_args())

    said = str(stopped.value)
    for way in ("MDCX_KEY", "--key-file", "--key"):
        assert way in said, f"the message does not mention {way}"


def test_the_help_warns_that_a_command_line_is_readable(capsys):
    """Relaxing --key without saying why would leave the same false comfort."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action")
    # Rebuilt the way main() does, to read what a user is told.
    old = sys.argv
    sys.argv = ["mdcx", "pack", "--help"]
    try:
        with pytest.raises(SystemExit):
            archive.main()
    finally:
        sys.argv = old

    printed = capsys.readouterr().out
    assert "process table" in printed, (
        "nothing tells the reader that --key is visible while it runs")
