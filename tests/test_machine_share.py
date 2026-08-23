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

# Machine sizes worth checking: the small ones where a fixed cap oversubscribes,
# and the large ones where reserving a single processor reserves nothing.
TAMANOS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128)


def _tope(procesadores: int, monkeypatch) -> int:
    monkeypatch.setattr(os, "cpu_count", lambda: procesadores)
    return cli._default_max_cores()


@pytest.mark.parametrize("procesadores", TAMANOS)
def test_a_share_of_the_machine_is_left_free(procesadores, monkeypatch):
    """Never more than what the reserve allows, at any size."""
    tope = _tope(procesadores, monkeypatch)
    permitido = int(procesadores * (1.0 - cli.RESERVED_SHARE))
    assert tope <= max(1, permitido), (
        f"on {procesadores} processors the default asks for {tope}, "
        f"which leaves less than {cli.RESERVED_SHARE:.0%} free")


@pytest.mark.parametrize("procesadores", TAMANOS)
def test_the_default_never_asks_for_more_than_the_machine_has(procesadores, monkeypatch):
    """The defect the constant had: 8 on a machine of four."""
    tope = _tope(procesadores, monkeypatch)
    assert 1 <= tope <= procesadores, (
        f"on {procesadores} processors the default asks for {tope}")


@pytest.mark.parametrize("procesadores", [n for n in TAMANOS if n >= 8])
def test_a_larger_machine_is_actually_used(procesadores, monkeypatch):
    """The cap grows with the machine; a constant is a ceiling nobody asked for.

    Reserving a share and then capping the rest at a fixed number leaves most of
    a large machine idle, which is the same defect in the other direction: the
    number stops describing the machine it is running on.
    """
    tope = _tope(procesadores, monkeypatch)
    assert tope >= procesadores // 2, (
        f"on {procesadores} processors the default asks for only {tope}")


def test_the_cap_grows_with_the_machine(monkeypatch):
    """Twice the machine, near enough twice the processes."""
    assert _tope(64, monkeypatch) > _tope(32, monkeypatch) > _tope(16, monkeypatch)


def test_an_unknown_processor_count_does_not_raise(monkeypatch):
    """os.cpu_count returns None where the platform will not say."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert cli._default_max_cores() >= 1


def _tamanos(monkeypatch, total, fraccion, vram_libre, tiene_gpu=True):
    monkeypatch.setattr(cli, "_free_vram_mib", lambda: vram_libre)
    monkeypatch.setattr(cli, "_vram_per_worker_mib", lambda: 1384)
    return cli._lane_sizes(total, fraccion, tiene_gpu)


def test_the_card_decides_when_it_is_the_scarce_one(monkeypatch):
    """Many processors and little video memory: the case worst served before.

    Raising the count by hand does the opposite of what it looks like there --
    twelve workers on a 6 GB card ask for 15.7 GB and measured three times
    slower than three -- because the resource running out was not the one being
    raised.
    """
    gpu, cpu = _tamanos(monkeypatch, total=12, fraccion=0.20, vram_libre=2000)
    assert gpu == 1, "more workers were put on the card than fit on it"
    assert cpu == 11


def test_the_work_decides_when_little_of_it_needs_the_card(monkeypatch):
    """A lane sized for documents that never arrive is a lane of idle processes."""
    gpu, cpu = _tamanos(monkeypatch, total=9, fraccion=0.20, vram_libre=11000)
    assert gpu == 2, "the card would allow more, but the work does not ask for it"
    assert cpu == 7


def test_more_of_the_work_needing_the_card_moves_workers_to_it(monkeypatch):
    """Measured on real material, the share ranges from 13% to 58%."""
    pocos, _ = _tamanos(monkeypatch, total=9, fraccion=0.13, vram_libre=5105)
    muchos, _ = _tamanos(monkeypatch, total=9, fraccion=0.58, vram_libre=5105)
    assert muchos > pocos, "the split does not follow what the material asks for"


def test_something_is_always_left_for_the_processor(monkeypatch):
    """Without this ceiling the formula breaks where there is most to gain.

    A card of 11 GB would otherwise take seven workers and leave one for
    everything else, which is backwards: the bulk of this work is processor
    work.
    """
    gpu, cpu = _tamanos(monkeypatch, total=8, fraccion=1.0, vram_libre=24000)
    assert cpu >= 1
    assert gpu <= 7, "every worker was put on the card"


def test_the_split_grows_with_the_machine(monkeypatch):
    """It used to be the constants 8 and 2, whatever the machine."""
    chica = _tamanos(monkeypatch, total=3, fraccion=0.20, vram_libre=24000)
    grande = _tamanos(monkeypatch, total=25, fraccion=0.20, vram_libre=24000)
    assert sum(grande) > sum(chica)
    assert grande[0] > chica[0], "the card lane did not grow with the machine"


def test_a_machine_with_no_card_is_not_measured_against_one(monkeypatch):
    """There is no video memory to divide, and the lane is still worth having."""
    gpu, cpu = _tamanos(monkeypatch, total=9, fraccion=0.30, vram_libre=None,
                        tiene_gpu=False)
    assert gpu >= 1 and cpu >= 1


def _lote(monkeypatch, libre_mib=None, hay_placa=True):
    """The batch this machine would choose, with the card it is told it has."""
    import types

    from mdcx.convert import tatr

    monkeypatch.delenv("MDCX_TATR_BATCH", raising=False)
    falso = types.ModuleType("torch")
    falso.cuda = types.SimpleNamespace(
        is_available=lambda: hay_placa,
        mem_get_info=lambda: ((libre_mib or 0) * 1024 * 1024, 0))
    monkeypatch.setitem(sys.modules, "torch", falso)
    return tatr._batch_size()


def test_a_worker_alone_does_not_size_its_own_batch(monkeypatch):
    """It cannot: what is free is free for every worker that may reach the model.

    Sizing it from the free memory looks careful and is not. Done that way on a
    6 GB card, three workers each took a batch of seventeen -- some 2,756 MiB
    apiece against a budget of 1,384 -- and the card reached 96% at full
    utilisation, which is the state the batching was meant to avoid.
    """
    from mdcx.convert import tatr

    assert _lote(monkeypatch, libre_mib=24000) == tatr.BATCH_FLOOR, (
        "a worker sized the batch from memory it does not have to itself")


def test_the_batch_fits_once_every_worker_is_seated(monkeypatch):
    """Whoever can see the whole run decides, and what it decides has to fit."""
    from mdcx.convert import tatr

    for libre, procesos in ((5113, 3), (5113, 1), (2000, 1), (11000, 4),
                            (24000, 5), (80000, 8), (1000, 2)):
        lote = tatr.batch_for(libre, procesos)
        pedido = procesos * (tatr.MODELS_MIB + lote * tatr.BATCH_MIB)
        asentados = procesos * (tatr.MODELS_MIB + tatr.BATCH_FLOOR * tatr.BATCH_MIB)
        if asentados <= libre:
            assert pedido <= libre, (
                f"{procesos} workers with a batch of {lote} ask for {pedido} MiB "
                f"of the {libre} free")


def test_room_left_over_buys_a_larger_batch(monkeypatch):
    """The cost of a page falls with the batch: 150 ms sending one, 46 with 24."""
    from mdcx.convert import tatr

    apretado = tatr.batch_for(5113, 3)
    holgado = tatr.batch_for(5113, 1)
    assert holgado > apretado, "the room left by fewer workers bought nothing"
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

    for libre in (8000, 24000, 80000, 200000):
        assert tatr.batch_for(libre, 1) <= tatr.BATCH_CEILING


def test_a_machine_with_no_card_keeps_the_floor(monkeypatch):
    """Nothing to read, and nothing gained by a larger batch either."""
    from mdcx.convert import tatr

    assert _lote(monkeypatch, hay_placa=False) == tatr.BATCH_FLOOR


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
    opciones = engines._accelerator_options()
    if opciones is None:
        pytest.skip("needs the convert extra: pip install 'mdcx[convert]'")
    assert opciones.num_threads == 1

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
    class Contador:
        def __init__(self):
            self.dentro = 0
            self.maximo = 0

        def acquire(self):
            self.dentro += 1
            self.maximo = max(self.maximo, self.dentro)

        def release(self):
            self.dentro -= 1

    puerta = Contador()
    engines.set_gpu_gate(puerta)
    try:
        with engines.gpu_turn():
            assert puerta.dentro == 1, "the model call did not take its turn"
        assert puerta.dentro == 0, "the turn was not given back"
        assert puerta.maximo == 1
    finally:
        engines.set_gpu_gate(None)


def test_the_turn_is_given_back_when_the_model_fails(monkeypatch):
    """A model that raises must not leave the card reserved for the whole run."""
    class Contador:
        def __init__(self):
            self.dentro = 0

        def acquire(self):
            self.dentro += 1

        def release(self):
            self.dentro -= 1

    puerta = Contador()
    engines.set_gpu_gate(puerta)
    try:
        with pytest.raises(RuntimeError):
            with engines.gpu_turn():
                raise RuntimeError("el modelo fallo")
        assert puerta.dentro == 0, (
            "a failure left the card held, and every other process waiting")
    finally:
        engines.set_gpu_gate(None)
