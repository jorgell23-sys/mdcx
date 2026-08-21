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

"""Output that cannot stop the work.

A console has a code page, and on Windows it is usually one that cannot
represent every character a document name may contain. Printing the name of a
file called `Fizyka dla szkół wyższych — łódź.pdf` then raises, and a conversion
that had succeeded is reported as a crash: what failed was showing the work, not
doing it.

This is not an unusual case. Of 195 files taken from open-access catalogues,
24 per cent had names outside ASCII, in Polish, Portuguese, German and Czech.

Two measures, because either alone leaves a gap. The stream is set to UTF-8,
which resolves it wherever the stream can be reconfigured, and printing falls
back to a representable form where it cannot -- a stream replaced by something
that has no reconfigure, which is what a wrapper or a test harness installs.
"""
from __future__ import annotations

import sys


def configure() -> None:
    """Set the standard streams to UTF-8, replacing what cannot be encoded.

    Called by every entry point. Doing it in one place rather than at each of
    them is what keeps a new entry point from being the one that lacks it,
    which is how this defect arrived: two commands configured their output and
    the third, the one that converts, did not.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a stream that cannot be reconfigured
            pass


def safe_print(message: str, **kwargs) -> None:
    """Print, degrading the message rather than raising.

    Reporting progress must never end the work it reports on. When the stream
    cannot represent a character, the name is written with that character
    replaced, which keeps the line readable and the run alive.
    """
    try:
        print(message, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, "replace").decode(encoding, "replace"),
              **kwargs)
