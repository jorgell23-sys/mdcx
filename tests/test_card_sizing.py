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

"""Who may use the card, and how the run finds out it guessed wrong.

Three faults reported together, and they compound. The card was declared
available where no device was visible; the lane meant to hold the card's work
received nothing while a quarter of the documents used the card anyway; and the
gate that bounds card use was sized by that empty lane, so everything which did
use the card queued behind one permit.

The measured consequence of the last one is the one worth keeping: running
mdcx's own orchestration was half the speed of invoking it once per document
from outside, on the same machine and the same books. The recommended way to use
the tool was the slow way.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import cli  # noqa: E402
from mdcx.convert import paths  # noqa: E402


# --- A card that is not there -------------------------------------------------


def test_no_visible_device_is_no_card(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=""` -- the empty string, which is the most
    widespread way of turning CUDA off -- leaves `is_available()` returning True
    with `device_count()` at zero.

    mdcx then named the card in its output and took the CUDA branch, so someone
    who believed they had freed the card had not. What that cost, measured by
    whoever reported it: hours at 100 per cent, ending in
    VIDEO_SCHEDULER_INTERNAL_ERROR and a reboot.
    """
    from mdcx.convert import engines

    class Pretending:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 0

            @staticmethod
            def get_device_name(i):
                return "Quadro RTX 3000"

    monkeypatch.setitem(sys.modules, "torch", Pretending)
    monkeypatch.setattr(engines, "_DEVICE_CACHE", {})

    assert engines.gpu_available() is False
    assert engines.device_name() == "CPU"


def test_a_visible_device_is_still_a_card(monkeypatch):
    """The count is the stricter condition, so nothing that works stops."""
    from mdcx.convert import engines

    class Present:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def get_device_name(i):
                return "Quadro RTX 3000"

    monkeypatch.setitem(sys.modules, "torch", Present)
    monkeypatch.setattr(engines, "_DEVICE_CACHE", {})

    assert engines.gpu_available() is True
    assert engines.device_name() == "Quadro RTX 3000"


# --- Mathematics does not announce a table ------------------------------------


def test_the_private_use_area_is_counted_across_its_three_planes():
    """Where a TeX font puts its delimiters, and therefore where mathematics
    set in two dimensions shows up in extracted text."""
    assert paths._private_use("ordinary prose") == 0
    assert paths._private_use(chr(0xE000) + chr(0xF8FF)) == 2
    assert paths._private_use(chr(0xF0000) + chr(0x100000)) == 2


def test_the_cut_is_a_number_and_says_it_is_one():
    """Two means are not two ranges. The cut sits between 55 and 481 averages
    and is not claimed to separate the groups -- what makes it safe to try is
    that the run reports what reached the card against what was expected."""
    assert 55 / 20 < paths.PRIVATE_PER_PAGE < 481 / 5


# --- What actually reached the card -------------------------------------------


def test_the_engines_that_use_the_card_are_named():
    """What decides is which engine ran, not which lane the document waited in:
    the hybrid engine is reached from either."""
    assert "hibrido" in cli.CARD_ENGINES
    assert "nativo" not in cli.CARD_ENGINES


def test_a_mismatch_between_expected_and_reached_is_reportable():
    """The count mdcx printed was a prediction presented as a fact: `0
    document(s) expected to reach the card` while 7 of 26 did. The sizing is
    computed before anything runs, so it can be wrong; what it must not do is
    stay quiet about it."""
    records = [{"engine": "nativo"}] * 19 + [{"engine": "hibrido"}] * 7

    reached = sum(1 for r in records if r.get("engine") in cli.CARD_ENGINES)

    assert reached == 7
    assert reached != 0, "the expectation this run was sized on"


# --- The gate is the card's, not a lane's -------------------------------------


def test_the_gate_is_not_sized_by_the_lane():
    """The fault, in one line of source.

    The gate applies to every worker in either lane -- the hybrid engine reaches
    the model from the processor lane -- and it was sized by the lane that,
    once correctly emptied, collapsed to one permit. Every document that used
    the card then queued behind that one.
    """
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "cli.py").read_text(encoding="utf-8")

    assert "Semaphore(card_gate)" in source
    assert "Semaphore(gpu_workers)" not in source, (
        "the card's gate is sized by a lane again")


def test_an_empty_lane_asks_for_no_processes():
    """They would pay their imports and, if they touch anything, the models --
    some sixteen seconds each -- to receive no document at all, while the cores
    they hold are the ones the other lane is short of."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "cli.py").read_text(encoding="utf-8")

    assert "if not lanes[LANE_GPU] and args.gpu_workers is None:" in source
    # And asking a pool for zero of them raises rather than meaning none.
    assert "max_workers=max(1, gpu_workers)" in source


# --- The fixed cost is per process, and nothing said so -----------------------


def test_documents_can_be_converted_one_at_a_time_in_one_process(tmp_path):
    """The retirement condition asked for a way to convert N documents without
    paying N times for the models, *keeping per-document control*.

    Handing a whole folder to mdcx-convert amortises the cost and takes away the
    order, the per-document reaction and the failure isolation a queue is for.
    This keeps both.
    """
    from mdcx.convert import resident

    for i in range(3):
        (tmp_path / f"n{i}.txt").write_text(f"# Doc {i}\n\nText {i}.\n",
                                            encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    seen = []
    with resident.warm(report=seen.append) as convert:
        records = [convert(tmp_path / f"n{i}.txt", out) for i in range(3)]

    assert len(seen) == 1, "the startup was not reported once"
    assert all(r["ok"] for r in records)
    assert sorted(p.name for p in out.rglob("*.md")) == ["n0.md", "n1.md", "n2.md"]


def test_the_records_are_the_ones_the_converter_writes(tmp_path):
    """Whatever reads one reads the other, or this would be a second format."""
    from mdcx.convert import resident

    (tmp_path / "a.txt").write_text("# A\n\nText.\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    record = next(resident.convert_documents([tmp_path / "a.txt"], out))

    for field in ("source_pseudopath", "markdown_pseudopath", "engine",
                  "seconds", "ok"):
        assert field in record, f"the record is missing {field}"


def test_the_startup_is_measured_once_and_remembered():
    """The number is what tells a caller whether a batch is worth amortising
    over, so asking for it must not itself cost anything the second time."""
    from mdcx.convert import resident

    first = resident.cost_of_starting()
    second = resident.cost_of_starting()

    assert first == second
    assert first >= 0.0


def test_a_file_outside_the_root_is_still_converted(tmp_path):
    """A queue does not necessarily hand paths under one folder, and refusing
    them would make this useless for the caller it exists for."""
    from mdcx.convert import resident

    (tmp_path / "loose.txt").write_text("# L\n\nText.\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    job = resident.job_for(tmp_path / "loose.txt", input_root=elsewhere)

    assert job.rel_target == Path("loose.md")
