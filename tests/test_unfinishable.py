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

"""A document the engine never finishes must not take the run with it.

There is material the structured engine does not terminate on. Measured on a
real collection: two workers spent three hours and twenty minutes of processor
time on documents of two to eight A4 pages, while equivalent documents in the
same batch took six seconds natively and up to thirty-five through layout
analysis. About 4% of that collection behaved this way.

Nothing distinguishes them beforehand. Against sixty documents that convert
normally, the ones that hang have fewer pages (median 10 against 18), the same
megabytes, the same images per page and slightly more text. There is nothing to
screen for, so the only defence is to stop waiting.

Without one, a single such document holds its worker for the length of the run,
and as many of them as there are workers stop the conversion altogether: the
batch cannot close, because it waits for all of them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import cli  # noqa: E402
from mdcx.convert import convert as convert_module  # noqa: E402
from mdcx.convert import engines  # noqa: E402
from mdcx.convert.convert import DocumentTimeout, _time_limit  # noqa: E402
from mdcx.convert.paths import Job  # noqa: E402


def test_work_that_does_not_finish_is_given_up_on():
    """The defect itself, in miniature: a computation that never returns."""
    started = time.time()
    with pytest.raises(DocumentTimeout):
        with _time_limit(0.5):
            while True:
                sum(range(10_000))
    spent = time.time() - started

    assert spent < 15, (
        f"the limit did not interrupt the work; it ran {spent:.1f}s past 0.5")


def test_work_that_finishes_is_not_touched():
    """A limit that fires on healthy documents is worse than none."""
    with _time_limit(30):
        result = "done"
    assert result == "done"


def test_no_limit_means_no_limit():
    """Zero and None both wait indefinitely, and neither may raise."""
    for setting in (0, 0.0, None):
        with _time_limit(setting):
            pass


def test_the_timer_does_not_outlive_the_document():
    """A cancelled limit must not fire into whatever the worker does next.

    The exception is delivered asynchronously to the thread, so one left
    pending would surface inside an unrelated document and be recorded against
    it.
    """
    with _time_limit(0.05):
        pass
    time.sleep(0.3)
    # If the timer had fired late, this line -- or the assertion -- would carry
    # the exception instead.
    assert sum(range(1000)) == 499500


def _job(tmp_path: Path) -> Job:
    source = tmp_path / "doc.txt"
    source.write_text("some text\n", encoding="utf-8")
    return Job(source=source, rel_source=Path("doc.txt"),
               rel_target=Path("doc.md"), kind="text", size=9)


def test_a_document_that_times_out_is_recorded_and_the_worker_goes_on(
        tmp_path, monkeypatch):
    """What the run gets back: a record, not a dead worker.

    The worker must return rather than die. A killed worker takes the pool with
    it and loses the batch, which is the outcome the limit exists to avoid.
    """
    def never_finishes(*args, **kwargs):
        while True:
            sum(range(10_000))

    monkeypatch.setattr(convert_module, "convert_one", never_finishes)

    record = convert_module.convert_one_safe(
        (_job(tmp_path), tmp_path / "out", False, False, True, 0.5))

    assert record["ok"] is False
    assert record["verification"]["status"] == "timeout", (
        "a document that was still going is recorded as an error, which reads "
        "as something being wrong with the document")
    assert "DocumentTimeout" in record["errors"][0]


def test_a_healthy_document_is_unaffected_by_the_limit(tmp_path, monkeypatch):
    """The limit is a backstop, and must not show up in the ordinary path."""
    monkeypatch.setattr(convert_module, "convert_one",
                        lambda *a, **k: {"ok": True, "engine": "nativo"})

    record = convert_module.convert_one_safe(
        (_job(tmp_path), tmp_path / "out", False, False, True, 30))

    assert record == {"ok": True, "engine": "nativo"}


def test_the_default_leaves_room_for_a_slow_book():
    """Twenty minutes against a worst ordinary case of thirty-five seconds.

    Abandoning a good document costs the whole of its work; waiting too long
    for a bad one costs one worker for the excess. The default is sized for the
    first mistake being the expensive one.
    """
    assert cli.DOCUMENT_TIMEOUT_SECONDS >= 600, (
        "the default is tight enough to abandon a long book with many tables")


def test_the_status_has_a_label_of_its_own():
    """A reader deciding what to retry needs it apart from a failure."""
    from mdcx.convert import index

    assert "timeout" in index._STATUS_LABEL
    assert index._STATUS_LABEL["timeout"] != index._STATUS_LABEL["error"]


# --- A partial run says it was partial --------------------------------------
#
# The progress file has a contract a consumer depends on to decide whether it
# may delete a source PDF: present means the run was cut short, and holds
# exactly what finished. A run limited on purpose finishes correctly and still
# leaves work behind, so the file survives and reads as a failure. It cost a
# real batch: 62 chapters converted at 100% coverage, recorded as "converted 0".


def _convert(tmp_path, count, limit=None):
    source = tmp_path / "in"
    source.mkdir(exist_ok=True)
    for i in range(count):
        (source / f"doc{i}.txt").write_text(f"Document {i}\n\ntext\n",
                                            encoding="utf-8")
    out = tmp_path / f"out{limit or 'all'}"
    argv = ["mdcx-convert", "--input", str(source), "--output", str(out),
            "--max-cores", "2", "--serial"]
    if limit:
        argv += ["--limit", str(limit)]
    old = sys.argv
    sys.argv = argv
    try:
        cli.main()
    finally:
        sys.argv = old
    return out


def test_a_complete_run_removes_the_progress_file(tmp_path):
    """The contract the consumer relies on, which must keep holding."""
    out = _convert(tmp_path, 3)
    assert not (out / cli.PROGRESS_FILENAME).exists()


def test_a_limited_run_keeps_the_file_and_says_it_finished(tmp_path):
    """Kept, so a limited run that dies halfway can still be resumed -- and
    marked, so its presence is no longer read as a failure."""
    out = _convert(tmp_path, 5, limit=2)
    progress = out / cli.PROGRESS_FILENAME
    assert progress.exists(), "a limited run can still be resumed, so it is kept"

    lines = [json.loads(l) for l in progress.read_text(encoding="utf-8").splitlines() if l.strip()]
    closing = [l for l in lines if l.get("run") == "complete"]
    assert closing, (
        "nothing in the file distinguishes this from a run that was cut short")
    assert closing[-1]["limited_to"] == 2
    assert closing[-1]["of"] == 5


def test_the_mark_does_not_disturb_the_records_around_it(tmp_path):
    """Whoever reads the file for finished documents must not trip over it."""
    out = _convert(tmp_path, 5, limit=2)
    lines = [json.loads(l) for l in
             (out / cli.PROGRESS_FILENAME).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    documents = [l for l in lines if l.get("markdown_pseudopath")]

    assert len(documents) == 2, "the mark was counted as a document"
    assert all(d.get("run") is None for d in documents)


# --- A permit for the card outlives the block that took it ------------------
#
# The gate limits how many processes use the card at once, taken on entering
# the model call and returned on leaving it. That is correct while the block
# ends -- and it does not always: layout analysis can leave a thread in a
# blocking call and abandon it, and the document timeout can cut in between
# taking the permit and entering the body. Either way __exit__ never runs.
#
# With two permits on the measured machine, two such documents were enough:
# four workers waiting in acquire(), none of them holding anything, the card
# idle, and the run never advancing again. The rate had been falling from 1,152
# to 180 chapters an hour before anyone understood why.


@pytest.fixture
def gate(monkeypatch):
    """A gate of two permits, as the machine in the report had."""
    import multiprocessing

    semaphore = multiprocessing.Semaphore(2)
    monkeypatch.setattr(engines, "_GPU_GATE", semaphore)
    monkeypatch.setattr(engines, "_TURNS_HELD", 0)
    monkeypatch.setattr(engines, "TURN_WAIT_SECONDS", 2.0)
    yield semaphore
    engines.release_turns()


def test_a_permit_is_returned_when_the_block_ends(gate):
    """The ordinary path, which has to keep working."""
    with engines.gpu_turn():
        assert engines.turns_held() == 1
    assert engines.turns_held() == 0


def test_a_permit_survives_a_block_that_never_ends(gate):
    """The defect: two documents that do not leave the block, and the gate is
    empty for the rest of the run."""
    engines._Turn().__enter__()
    engines._Turn().__enter__()
    assert engines.turns_held() == 2, "premise: both permits are out"

    recovered = engines.release_turns()

    assert recovered == 2, "the permits could not be recovered"
    assert engines.turns_held() == 0
    # And the gate works again, which is what the run needs.
    with engines.gpu_turn():
        pass


def test_a_document_squares_up_its_permits(tmp_path, monkeypatch, gate):
    """The boundary that always ends is the document, so it is where they are
    returned -- including when the document is given up on."""
    def takes_a_permit_and_hangs(*args, **kwargs):
        engines._Turn().__enter__()      # taken, never returned
        while True:
            sum(range(10_000))

    monkeypatch.setattr(convert_module, "convert_one", takes_a_permit_and_hangs)

    record = convert_module.convert_one_safe(
        (_job(tmp_path), tmp_path / "out", False, False, True, 0.5))

    assert record["verification"]["status"] == "timeout"
    assert engines.turns_held() == 0, (
        "the abandoned document kept its permit, which is how the gate emptied")


def test_an_exhausted_gate_does_not_stop_the_run(gate):
    """The belt to the braces: even if a permit were lost beyond recovery.

    Waiting for a turn that will not come is certain death; going ahead without
    one risks contention. The report measured the first.
    """
    gate.acquire()
    gate.acquire()                       # both gone, and not ours to return
    assert engines.turns_held() == 0

    started = time.time()
    with engines.gpu_turn():
        pass
    waited = time.time() - started

    assert waited < 30, "the run would have waited for a turn that never comes"
    assert engines.turns_held() == 0, "it invented a permit it does not hold"
    gate.release()
    gate.release()


def test_returning_is_never_more_than_was_taken(gate):
    """Handing back a permit twice would raise the ceiling for the whole run."""
    with engines.gpu_turn():
        pass
    assert engines.release_turns() == 0

    for _ in range(5):
        engines.release_turns()

    # The gate still holds exactly two, which is what it was built with.
    assert gate.acquire(timeout=1)
    assert gate.acquire(timeout=1)
    assert not gate.acquire(timeout=0.2), "the gate now allows more than it should"
    gate.release()
    gate.release()


def test_workers_are_replaced_so_an_abandoned_thread_frees_its_core():
    """A thread cannot be cancelled, so the process holding it is replaced."""
    assert cli.RECYCLE_AFTER_DOCUMENTS > 0
    assert cli.RECYCLE_AFTER_DOCUMENTS <= 200, (
        "a lifetime this long leaves a burnt core in place for most of a run")
