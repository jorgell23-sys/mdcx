"""Checks that a conversion leaves part of the machine to whoever is using it.

Converting a library is the heaviest thing this package does, and it runs on
the machine somebody is working at. The rule is a share rather than a count:
one free processor is a quarter of a machine of four and nothing at all on one
of thirty-two, and it is the large machine where taking everything is worst.

Two things used to spend that share without saying so.

The cap was the bare constant 8, which asks for twice the machine on one of
four processors. It went unnoticed for a while because almost every document
queued in a lane of three regardless of the cap, so raising or lowering it
changed little; once the lanes were filled it became what decides the real
concurrency.

And the cap counted processes, not threads. Docling asks for four threads of
its own, so eight workers asked for thirty-two threads on a machine of twelve,
and the reserve disappeared inside the pool.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import cli  # noqa: E402
from mdcx.convert import engines  # noqa: E402
from mdcx.convert import paths  # noqa: E402

# Machine sizes worth checking: the small ones where a fixed cap oversubscribes,
# and the large ones where reserving a single processor reserves nothing.
SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128)


def _cap(processors: int, monkeypatch) -> int:
    monkeypatch.setattr(os, "cpu_count", lambda: processors)
    return cli._default_max_cores()


@pytest.mark.parametrize("processors", SIZES)
def test_a_share_of_the_machine_is_left_free(processors, monkeypatch):
    """Never more than what the reserve allows, at any size."""
    cap = _cap(processors, monkeypatch)
    allowed = int(processors * (1.0 - cli.RESERVED_SHARE))
    assert cap <= max(1, allowed), (
        f"on {processors} processors the default asks for {cap}, "
        f"which leaves less than {cli.RESERVED_SHARE:.0%} free")


@pytest.mark.parametrize("processors", SIZES)
def test_the_default_never_asks_for_more_than_the_machine_has(processors, monkeypatch):
    """The defect the constant had: 8 on a machine of four."""
    cap = _cap(processors, monkeypatch)
    assert 1 <= cap <= processors, (
        f"on {processors} processors the default asks for {cap}")


@pytest.mark.parametrize("processors", [n for n in SIZES if n >= 8])
def test_a_larger_machine_is_actually_used(processors, monkeypatch):
    """The cap grows with the machine; a constant is a ceiling nobody asked for.

    Reserving a share and then capping the rest at a fixed number leaves most of
    a large machine idle, which is the same defect in the other direction: the
    number stops describing the machine it is running on.
    """
    cap = _cap(processors, monkeypatch)
    assert cap >= processors // 2, (
        f"on {processors} processors the default asks for only {cap}")


def test_the_cap_grows_with_the_machine(monkeypatch):
    """Twice the machine, near enough twice the processes."""
    assert _cap(64, monkeypatch) > _cap(32, monkeypatch) > _cap(16, monkeypatch)


def test_an_unknown_processor_count_does_not_raise(monkeypatch):
    """os.cpu_count returns None where the platform will not say."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert cli._default_max_cores() >= 1


def _sizes(monkeypatch, total, fraction, vram_free, has_gpu=True):
    monkeypatch.setattr(cli, "_free_vram_mib", lambda: vram_free)
    monkeypatch.setattr(cli, "_vram_per_worker_mib", lambda: 1384)
    return cli._lane_sizes(total, fraction, has_gpu)


def test_the_card_decides_when_it_is_the_scarce_one(monkeypatch):
    """Many processors and little video memory: the case worst served before.

    Raising the count by hand does the opposite of what it looks like there --
    twelve workers on a 6 GB card ask for 15.7 GB and measured three times
    slower than three -- because the resource running out was not the one being
    raised.
    """
    gpu, cpu = _sizes(monkeypatch, total=12, fraction=0.20, vram_free=2000)
    assert gpu == 1, "more workers were put on the card than fit on it"
    # The processor lane used to take the whole remainder here -- eleven
    # processes, every one of them loading models onto a card with 2 GB free.
    # That is the stall this sizing now prevents: what is left of the processor
    # count is not free when the processes it counts all reach the model.
    assert cpu < 11, "eleven processes would load models onto a 2 GB card"
    assert cpu >= 1


def test_the_work_decides_when_little_of_it_needs_the_card(monkeypatch):
    """A lane sized for documents that never arrive is a lane of idle processes."""
    gpu, cpu = _sizes(monkeypatch, total=9, fraction=0.20, vram_free=11000)
    assert gpu == 2, "the card would allow more, but the work does not ask for it"
    assert cpu == 7


def test_more_of_the_work_needing_the_card_moves_workers_to_it(monkeypatch):
    """Measured on real material, the share ranges from 13% to 58%."""
    few, _ = _sizes(monkeypatch, total=9, fraction=0.13, vram_free=5105)
    many, _ = _sizes(monkeypatch, total=9, fraction=0.58, vram_free=5105)
    assert many > few, "the split does not follow what the material asks for"


def test_something_is_always_left_for_the_processor(monkeypatch):
    """Without this ceiling the formula breaks where there is most to gain.

    A card of 11 GB would otherwise take seven workers and leave one for
    everything else, which is backwards: the bulk of this work is processor
    work.
    """
    gpu, cpu = _sizes(monkeypatch, total=8, fraction=1.0, vram_free=24000)
    assert cpu >= 1
    assert gpu <= 7, "every worker was put on the card"


def test_the_split_grows_with_the_machine(monkeypatch):
    """It used to be the constants 8 and 2, whatever the machine."""
    small = _sizes(monkeypatch, total=3, fraction=0.20, vram_free=24000)
    large = _sizes(monkeypatch, total=25, fraction=0.20, vram_free=24000)
    assert sum(large) > sum(small)
    assert large[0] > small[0], "the card lane did not grow with the machine"


def test_a_machine_with_no_card_is_not_measured_against_one(monkeypatch):
    """There is no video memory to divide, and the lane is still worth having."""
    gpu, cpu = _sizes(monkeypatch, total=9, fraction=0.30, vram_free=None,
                        has_gpu=False)
    assert gpu >= 1 and cpu >= 1


def _batch(monkeypatch, free_mib=None, gpu_present=True):
    """The batch this machine would choose, with the card it is told it has."""
    import types

    from mdcx.convert import tatr

    monkeypatch.delenv("MDCX_TATR_BATCH", raising=False)
    bogus = types.ModuleType("torch")
    bogus.cuda = types.SimpleNamespace(
        is_available=lambda: gpu_present,
        mem_get_info=lambda: ((free_mib or 0) * 1024 * 1024, 0))
    monkeypatch.setitem(sys.modules, "torch", bogus)
    return tatr._batch_size()


def test_a_worker_alone_does_not_size_its_own_batch(monkeypatch):
    """It cannot: what is free is free for every worker that may reach the model.

    Sizing it from the free memory looks careful and is not. Done that way on a
    6 GB card, three workers each took a batch of seventeen -- some 2,756 MiB
    apiece against a budget of 1,384 -- and the card reached 96% at full
    utilisation, which is the state the batching was meant to avoid.
    """
    from mdcx.convert import tatr

    assert _batch(monkeypatch, free_mib=24000) == tatr.BATCH_FLOOR, (
        "a worker sized the batch from memory it does not have to itself")


def test_the_batch_fits_once_every_worker_is_seated(monkeypatch):
    """Whoever can see the whole run decides, and what it decides has to fit."""
    from mdcx.convert import tatr

    for free, processes in ((5113, 3), (5113, 1), (2000, 1), (11000, 4),
                            (24000, 5), (80000, 8), (1000, 2)):
        batch = tatr.batch_for(free, processes)
        requested = processes * (tatr.MODELS_MIB + batch * tatr.BATCH_MIB)
        settled = processes * (tatr.MODELS_MIB + tatr.BATCH_FLOOR * tatr.BATCH_MIB)
        if settled <= free:
            assert requested <= free, (
                f"{processes} workers with a batch of {batch} ask for {requested} MiB "
                f"of the {free} free")


def test_room_left_over_buys_a_larger_batch(monkeypatch):
    """The cost of a page falls with the batch: 150 ms sending one, 46 with 24."""
    from mdcx.convert import tatr

    tight = tatr.batch_for(5113, 3)
    roomy = tatr.batch_for(5113, 1)
    assert roomy > tight, "the room left by fewer workers bought nothing"
    assert tatr.batch_for(80000, 2) == tatr.BATCH_CEILING


def test_a_card_with_no_room_left_keeps_the_floor(monkeypatch):
    """Nothing over means nothing to spend, and the floor is what was budgeted."""
    from mdcx.convert import tatr

    assert tatr.batch_for(1000, 4) == tatr.BATCH_FLOOR
    assert tatr.batch_for(None, 3) == tatr.BATCH_FLOOR
    assert tatr.batch_for(5113, 0) == tatr.BATCH_FLOOR


def test_the_batch_never_grows_past_where_the_curve_flattens(monkeypatch):
    """Past twenty-four the saving is small and the memory is not."""
    from mdcx.convert import tatr

    for free in (8000, 24000, 80000, 200000):
        assert tatr.batch_for(free, 1) <= tatr.BATCH_CEILING


def test_a_machine_with_no_card_keeps_the_floor(monkeypatch):
    """Nothing to read, and nothing gained by a larger batch either."""
    from mdcx.convert import tatr

    assert _batch(monkeypatch, gpu_present=False) == tatr.BATCH_FLOOR


def test_a_machine_can_say_otherwise(monkeypatch):
    """Thousands of cards, and some of them will measure differently."""
    from mdcx.convert import tatr

    monkeypatch.setenv("MDCX_TATR_BATCH", "3")
    assert tatr._batch_size() == 3


def test_the_threads_a_worker_asks_for_are_the_ones_it_was_given(monkeypatch):
    """The budget is divided among the workers, and Docling is told its share.

    Four apiece was a constant, so the count of processes said nothing about
    what the run would take.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    options = engines._accelerator_options()
    if options is None:
        pytest.skip("needs the convert extra: pip install 'mdcx[convert]'")
    assert options.num_threads == 1

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert engines._accelerator_options().num_threads == 3


def test_a_worker_budget_is_written_where_the_libraries_read_it(monkeypatch):
    """They read the environment when they load, so it is set before they do."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    try:
        cli._worker_budget(2, None)
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["MKL_NUM_THREADS"] == "2"
    finally:
        engines.set_gpu_gate(None)


def test_a_worker_with_no_gate_waits_for_nothing(monkeypatch):
    """One process has nothing to contend for, and neither does a test."""
    engines.set_gpu_gate(None)
    with engines.gpu_turn():
        pass  # it must simply not block


def test_the_card_is_bounded_by_a_gate_and_not_by_a_lane(monkeypatch):
    """Both lanes may use the card; how many at once is said once, here.

    The lane used to be the bound, by being small and by being the only lane
    whose workers were given the engines at all. That decided two separate
    things with one number, and the second was never wanted: it cost the CPU
    lane a third of its table rows and every list item.
    """
    class CallCounter:
        def __init__(self):
            self.dentro = 0
            self.maximo = 0

        def acquire(self, timeout=None):
            # Waited for in slices, so the wait can be interrupted by the
            # document timeout; a gate says whether it granted the turn.
            self.dentro += 1
            self.maximo = max(self.maximo, self.dentro)
            return True

        def release(self):
            self.dentro -= 1

    gate = CallCounter()
    engines.set_gpu_gate(gate)
    monkeypatch.setattr(engines, "_TURNS_HELD", 0)
    try:
        with engines.gpu_turn():
            assert gate.dentro == 1, "the model call did not take its turn"
        assert gate.dentro == 0, "the turn was not given back"
        assert gate.maximo == 1
    finally:
        engines.set_gpu_gate(None)


def test_the_turn_is_given_back_when_the_model_fails(monkeypatch):
    """A model that raises must not leave the card reserved for the whole run."""
    class CallCounter:
        def __init__(self):
            self.dentro = 0

        def acquire(self, timeout=None):
            self.dentro += 1
            return True

        def release(self):
            self.dentro -= 1

    gate = CallCounter()
    engines.set_gpu_gate(gate)
    monkeypatch.setattr(engines, "_TURNS_HELD", 0)
    try:
        with pytest.raises(RuntimeError):
            with engines.gpu_turn():
                raise RuntimeError("the model failed")
        assert gate.dentro == 0, (
            "a failure left the card held, and every other process waiting")
    finally:
        engines.set_gpu_gate(None)


# --- The demand for the card is not the size of the card's lane -------------
#
# Correcting classify_lane emptied the GPU lane, which was right: 197 documents
# to the processor lane and none to the card's. But the card's share was being
# computed from that same lane, so it fell to a single process -- while 55 of
# those 197 reached the model anyway, through the hybrid engine. Same output,
# 30% longer: 420 s against 307 s.


def _text_pdf(tmp_path, monkeypatch, page_text, pages=40):
    """A PDF with a real text layer, whose pages say what is passed in."""
    class Page:
        def __init__(self, text):
            self.text = text

    class Document:
        def __init__(self, pages):
            self.pages = pages

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, i):
            return self.pages[i]

        def close(self):
            pass

    from mdcx.convert import pdf as _pdf

    doc = Document([Page(page_text) for _ in range(pages)])
    monkeypatch.setattr(_pdf, "open_document", lambda path: doc)
    monkeypatch.setattr(_pdf, "page_text", lambda page: page.text)
    monkeypatch.setattr(_pdf, "count_images", lambda page: 0)

    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4\n")
    return paths.Job(source=book, rel_source=Path("book.pdf"),
                     rel_target=Path("book.md"), kind="pdf", size=9)


def test_a_document_read_from_its_text_layer_can_still_reach_the_card(
        tmp_path, monkeypatch):
    """The lane says processor and the model is used anyway.

    This is the whole of the defect. The page has a text layer, so it is not
    sent to optical recognition, and it announces a table, so the hybrid engine
    hands that page to the model -- which is on the card, from the processor
    lane.
    """
    body = "x" * 2700 + "\nTable 3.1: measurements\n"
    lane, reaches_card = paths.inspect_job(_text_pdf(tmp_path, monkeypatch, body))

    assert lane == paths.LANE_CPU, "a readable document does not need recognition"
    assert reaches_card, (
        "a document announcing a table was counted as having no use for the "
        "card, which is what left the card's share at one process")


def test_prose_with_no_tables_does_not_ask_for_the_card(tmp_path, monkeypatch):
    """The count has to discriminate, or it is the old constant again."""
    body = "x" * 2700 + "\nAs table 4 shows, the figure rises.\n"
    lane, reaches_card = paths.inspect_job(_text_pdf(tmp_path, monkeypatch, body))

    assert lane == paths.LANE_CPU
    assert not reaches_card, (
        "a sentence mentioning a table was read as a table caption")


def test_a_document_with_no_text_layer_still_asks_for_the_card(
        tmp_path, monkeypatch):
    """Optical recognition is the one thing that genuinely wants the card."""
    lane, reaches_card = paths.inspect_job(_text_pdf(tmp_path, monkeypatch, ""))

    assert lane == paths.LANE_GPU
    assert reaches_card


def test_sizing_the_card_by_its_lane_starves_it(monkeypatch):
    """The two formulas on the measured run, side by side.

    197 documents, none in the card's lane, 55 of them reaching the model. The
    lane says nothing needs the card; the work says a quarter of it does.
    """
    by_lane, _ = _sizes(monkeypatch, total=8, fraction=0.0, vram_free=11000)
    by_work, _ = _sizes(monkeypatch, total=8, fraction=55 / 197, vram_free=11000)

    assert by_lane == 1, "the empty lane no longer collapses the share"
    assert by_work >= 2, (
        "the work that actually reaches the card still sizes it at one process")


def test_an_empty_card_lane_does_not_mean_an_idle_card(tmp_path, monkeypatch):
    """The measured run, in miniature: nothing in the card's lane, work on it.

    Every document is readable, so classify_lane sends all of them to the
    processor -- which is correct, and was the point of the previous fix. A
    quarter of them announce tables, so the hybrid engine will put those on the
    card. The share has to come from the second number, not the first.
    """
    class Page:
        def __init__(self, text):
            self.text = text

    class Document:
        def __init__(self, pages):
            self.pages = pages

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, i):
            return self.pages[i]

        def close(self):
            pass

    prose = "x" * 2700
    with_table = prose + "\nTable 3.1: measurements\n"

    from mdcx.convert import pdf as _pdf

    bodies: dict[Path, str] = {}
    monkeypatch.setattr(_pdf, "open_document",
                        lambda path: Document([Page(bodies[Path(path)])] * 40))
    monkeypatch.setattr(_pdf, "page_text", lambda page: page.text)
    monkeypatch.setattr(_pdf, "count_images", lambda page: 0)

    pending = []
    for i in range(197):
        book = tmp_path / f"book{i}.pdf"
        book.write_bytes(b"%PDF-1.4\n")
        bodies[book] = with_table if i < 55 else prose
        pending.append(paths.Job(source=book, rel_source=Path(f"book{i}.pdf"),
                                 rel_target=Path(f"book{i}.md"), kind="pdf",
                                 size=9))

    lanes, card_bound = cli._dispatch(pending, use_docling=True)

    assert not lanes[paths.LANE_GPU], "the card's lane holds what needs recognition"
    assert card_bound == 55, "the work that reaches the card was miscounted"

    gpu, cpu = _sizes(monkeypatch, total=8,
                      fraction=card_bound / len(pending), vram_free=11000)
    assert gpu >= 2, (
        "the card was left with a single process while a quarter of the work "
        "was queueing for it")
    assert cpu >= 1


# --- What is resident on the card, not only what is computing ---------------
#
# The gate bounds how many processes may use the model at once. It says nothing
# about how many may load it, and a process pays for the models when it builds
# the converter, before it ever asks for a turn. Since the hybrid engine runs in
# the processor lane, every process there ends up holding a set of models.
#
# Measured on a 6 GB card: seven resident filled it and the run stopped dead --
# 76 chapters in 109 s, then none in 58 minutes -- while five finished the same
# work and were faster than six.

CARD_MIB = 6144
FREE_MIB = CARD_MIB - 257        # what the desktop was already using


def _sizes_resident(monkeypatch, total, fraction, free=FREE_MIB):
    monkeypatch.setattr(cli, "_free_vram_mib", lambda: free)
    monkeypatch.setattr(cli, "_vram_per_worker_mib", lambda: 1384)
    return cli._lane_sizes(total, fraction, True)


def test_the_processor_lane_is_bounded_by_what_the_card_can_hold(monkeypatch):
    """The run that stopped: 325 chapters, few of them needing the card.

    The processor lane used to be simply the remainder of the cap, so it took
    seven processes and all seven loaded models. This is the defect, and the
    lane must now come out smaller than the leftover.
    """
    gpu, cpu = _sizes_resident(monkeypatch, total=9, fraction=57 / 325)

    assert cpu < 9 - gpu or gpu + cpu < 9, (
        "the processor lane is still just the remainder of the processor count")
    assert cpu <= 6, (
        f"{cpu} processes would hold models on a 6 GB card; seven stopped the run")


def test_easy_material_is_the_likeliest_to_stall(monkeypatch):
    """The reversal worth a test: less card demand, more processes loading.

    Because the processor lane is the remainder, a low card fraction gives the
    most processes -- and the fewest turns. The bound has to hold precisely
    there, which is where nothing looks wrong.
    """
    _, cpu_easy = _sizes_resident(monkeypatch, total=12, fraction=0.05)
    _, cpu_heavy = _sizes_resident(monkeypatch, total=12, fraction=0.90)

    assert cpu_easy <= 6, (
        f"material that barely needs the card put {cpu_easy} processes on it")
    assert cpu_heavy >= 1


def test_the_configuration_that_stopped_is_no_longer_reachable(monkeypatch):
    """`--max-cores 12` with three turns gave nine resident, and it stopped."""
    gpu, cpu = _sizes_resident(monkeypatch, total=12, fraction=134 / 153)

    assert cpu <= 6, f"{cpu} resident processes is the configuration that stalled"
    assert gpu >= 1 and cpu >= 1


def test_a_bigger_card_is_allowed_more_resident(monkeypatch):
    """The bound is the card's, not a constant: 24 GB should hold more than 6."""
    _, small = _sizes_resident(monkeypatch, total=16, fraction=0.20, free=6000)
    _, large = _sizes_resident(monkeypatch, total=16, fraction=0.20, free=24000)

    assert large > small, "the resident bound does not follow the card"


def test_a_machine_with_no_card_keeps_the_whole_remainder(monkeypatch):
    """Nothing is resident where there is nothing to be resident on."""
    monkeypatch.setattr(cli, "_free_vram_mib", lambda: None)
    gpu, cpu = cli._lane_sizes(9, 0.20, False)

    assert gpu + cpu == 9, "processors were withheld for a card that is not there"


def test_the_resident_bound_leaves_a_margin(monkeypatch):
    """Filling the card is not using it: 96% stopped, 88% finished and was faster."""
    monkeypatch.setattr(cli, "_vram_per_worker_mib", lambda: 1384)
    gpu = 3
    cap = cli._resident_cap(FREE_MIB, gpu)
    committed = gpu * 1384 + max(0, cap - gpu) * cli.VRAM_RESIDENT_MIB

    assert committed <= FREE_MIB * 0.90, (
        f"the sizing commits {committed} of {FREE_MIB} MiB, which is the "
        "fraction at which the measured run stopped advancing")
