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

"""A server whose client has gone should not still be here.

Ten of them were measured alive at once, the oldest for two and a half hours,
retaining 7.25 GB between them -- and the shape is the tell: no processor at
all, not a loop, not a request being served. Just there.

Returning from `main()` is not the same as exiting. Python waits for every
non-daemon thread before it shuts down, and the libraries under an encoder start
some. Whichever of the candidate causes it was -- and it was not proven, by them
or here -- a server with no client has nothing left to finish, so it leaves.

And it lets go of the encoder long before that, because the damage does not
depend on the cause: three gigabytes held by a process answering nobody is worth
avoiding even if the process is entitled to be alive.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import mcp_server  # noqa: E402
from mdcx import semantic as S  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                          capture_output=True, text=True, timeout=60,
                          env={"PATH": "", "PYTHONPATH": SRC,
                               "SYSTEMROOT": r"C:\Windows"})


def test_the_process_ends_even_with_a_thread_that_would_hold_it():
    """The condition under the defect, reproduced.

    A non-daemon thread is enough to keep a finished interpreter alive
    indefinitely. Without `_leave` this subprocess would sit there until the
    timeout; with it, it is gone.
    """
    done = _run("""
        import threading, time, sys
        sys.path.insert(0, %r)
        from mdcx import mcp_server

        # Exactly what an encoder's libraries leave behind: a thread nobody
        # will join and nothing will interrupt.
        threading.Thread(target=lambda: time.sleep(600), daemon=False).start()
        mcp_server._leave(0)
    """ % SRC)

    assert done.returncode == 0


def test_it_carries_the_code_out():
    assert _run("""
        import sys
        sys.path.insert(0, %r)
        from mdcx import mcp_server
        mcp_server._leave(3)
    """ % SRC).returncode == 3


def test_the_watcher_cannot_be_what_keeps_the_process_alive():
    """A repair that installed a non-daemon thread would be causing the very
    thing it set out to fix."""
    import threading

    before = {t.name for t in threading.enumerate()}
    mcp_server._watch_for_idleness()
    started = [t for t in threading.enumerate() if t.name not in before]

    if mcp_server.IDLE_MINUTES > 0:
        assert started, "the watcher did not start"
        assert all(t.daemon for t in started)


def test_releasing_the_encoder_is_reversible():
    """The cost of letting go is seconds on the next question, and that is the
    whole trade: three gigabytes against a reload."""
    if not S.available():
        pytest.skip("no encoder installed")

    S.load()
    assert S.loaded()
    assert S.unload() is True
    assert not S.loaded()
    assert S.unload() is False, "letting go twice is not an error, but it is not a release"
    assert S.load() is not None


def test_the_idle_release_can_be_turned_off():
    """A long silence is not always idleness -- someone may be reading -- so
    whoever knows their own usage can say so."""
    import importlib
    import os

    original = os.environ.get("MDCX_IDLE_UNLOAD_MINUTES")
    try:
        os.environ["MDCX_IDLE_UNLOAD_MINUTES"] = "0"
        importlib.reload(mcp_server)
        assert mcp_server.IDLE_MINUTES == 0
        # And with it off, nothing is started at all.
        import threading
        before = {t.name for t in threading.enumerate()}
        mcp_server._watch_for_idleness()
        assert {t.name for t in threading.enumerate()} == before
    finally:
        if original is None:
            os.environ.pop("MDCX_IDLE_UNLOAD_MINUTES", None)
        else:
            os.environ["MDCX_IDLE_UNLOAD_MINUTES"] = original
        importlib.reload(mcp_server)


def test_asking_something_counts_as_not_idle():
    """Otherwise the encoder would be dropped from under a working client."""
    mcp_server._asked_now()
    first = mcp_server._LAST_ASKED[0]
    mcp_server._asked_now()

    assert mcp_server._LAST_ASKED[0] >= first
