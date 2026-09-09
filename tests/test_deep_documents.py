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

"""Documents discarded for a constant of the interpreter rather than a defect.

Measured by a consumer over 119,235 mathematics PDFs: two in a hundred thousand
came out with no Markdown at all and `RecursionError` in the record. The same
files, in the same process, converted whole -- 190,072 characters and no errors
-- once the recursion limit and the thread stack were raised. Nothing was wrong
with the documents.

Two in a hundred thousand is a small share and whole works, and the way it
presented was the problem: `RecursionError` reads as a broken file, so the
material was thrown away by whoever read the record.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import convert as convert_module  # noqa: E402


def _failed_on_depth():
    return {"engine": "none", "ok": False,
            "errors": ["nativo: ArgumentError: argument 1: RecursionError: "
                       "maximum recursion depth exceeded"]}


def _converted():
    return {"engine": "nativo", "ok": True, "errors": []}


# --- Telling the environment from the document ---------------------------------


def test_a_recursion_failure_is_recognised():
    assert convert_module._depth_was_the_limit(_failed_on_depth()) is True


def test_a_document_that_converted_is_not_a_depth_failure():
    """A conversion that produced something is not retried whatever its
    warnings say, or every run would convert twice."""
    record = _converted()
    record["errors"] = ["RecursionError somewhere in a warning"]

    assert convert_module._depth_was_the_limit(record) is False


def test_another_failure_is_not_mistaken_for_this_one():
    assert convert_module._depth_was_the_limit(
        {"engine": "none", "errors": ["MemoryError"]}) is False


# --- Room to recurse, and only where it is needed ------------------------------


def test_the_retry_gets_the_depth_and_the_caller_keeps_its_own():
    """Both numbers, and neither leaks. Raising the limit without the stack
    turns a catchable exception into a stack overflow, which in a conversion
    pool takes the lane's batch down with the process."""
    before = sys.getrecursionlimit()

    inside = convert_module._with_deep_stack(sys.getrecursionlimit)

    assert inside == convert_module.DEEP_RECURSION_LIMIT
    assert inside > before
    assert sys.getrecursionlimit() == before, "the caller's interpreter changed"


def test_the_stack_is_asked_for_in_the_same_breath():
    """The stack size can only be set before a thread is made, which is why
    the retry is a thread at all rather than a call in place."""
    assert convert_module.DEEP_STACK_BYTES >= 32 * 1024 * 1024


def test_an_error_from_the_retry_reaches_the_caller():
    """Running in a thread must not swallow what the thread raised."""
    def raises():
        raise ValueError("from inside")

    try:
        convert_module._with_deep_stack(raises)
    except ValueError as exc:
        assert str(exc) == "from inside"
    else:
        raise AssertionError("the exception was swallowed by the thread")


# --- What convert_one does with it ---------------------------------------------


def test_a_deep_document_is_converted_on_the_second_attempt(monkeypatch):
    """The recovery, which is the whole point: material that was being
    discarded as broken comes out whole."""
    attempts = []

    def attempt(*args, **kwargs):
        attempts.append(1)
        return _failed_on_depth() if len(attempts) == 1 else _converted()

    monkeypatch.setattr(convert_module, "_convert_one", attempt)

    record = convert_module.convert_one(object(), Path("out"))

    assert len(attempts) == 2
    assert record["ok"] is True
    assert record["recovered"] == "recursion depth"


def test_a_document_that_converts_is_not_converted_twice(monkeypatch):
    """The retry is reached only from a record that already failed, so normal
    material pays nothing for it."""
    attempts = []

    def attempt(*args, **kwargs):
        attempts.append(1)
        return _converted()

    monkeypatch.setattr(convert_module, "_convert_one", attempt)

    convert_module.convert_one(object(), Path("out"))

    assert len(attempts) == 1


def test_a_document_still_too_deep_says_so_in_a_field(monkeypatch):
    """A consumer deciding what to retry cannot act on a traceback. One is
    discarded and the other is retried elsewhere, and the record has to say
    which -- `RecursionError` alone reads as a file to throw away."""
    monkeypatch.setattr(convert_module, "_convert_one",
                        lambda *a, **k: _failed_on_depth())

    record = convert_module.convert_one(object(), Path("out"))

    assert record["failure"] == "environment"
    assert any("deeper than the interpreter allows" in e
               for e in record["errors"])


def test_a_platform_without_a_bigger_stack_keeps_the_first_failure(monkeypatch):
    """Retrying without the stack would trade a caught exception for a dead
    process, so the document keeps the failure it already has."""
    monkeypatch.setattr(convert_module, "_convert_one",
                        lambda *a, **k: _failed_on_depth())
    monkeypatch.setattr(convert_module, "_with_deep_stack", lambda call: None)

    record = convert_module.convert_one(object(), Path("out"))

    assert record["failure"] == "environment"
    assert record["engine"] == "none"
