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
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cli._worker_budget(2, True)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ, (
        "the GPU lane is the lane that keeps the card")

    cli._worker_budget(2, False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "", (
        "a CPU-lane worker must not reserve video memory")
