"""Checks how the encoder spends the accelerator, and that it stays correct.

Encoding is what publishing a corpus costs: thousands of seconds against tens
for compression and a fraction of one for encryption. It is also the part with
no failure to point at -- the vectors are right whatever the batching, so
nothing here shows up as a broken test. What shows up is a corpus that takes a
week instead of a day.

Two things decide the rate. Reduced precision, which the accelerator has
hardware for and a processor does not, so it is asked for per device rather than
set once. And the size of a batch, which costs its rows times its longest
column: filling one to a budget keeps that product flat, where a fixed count of
texts lets a single long passage set the width for everyone.

What these tests guard is that the ordering the budget relies on never reaches
the caller: the vectors must come back in the order the texts went in. Nothing
downstream would notice if they did not. Every passage would simply carry
another passage's meaning, and the package would be quietly wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import semantic  # noqa: E402

necesita_modelo = pytest.mark.skipif(
    not semantic.available(),
    reason="needs the multilingual extra: pip install 'mdcx[multilingual]'")


def test_reduced_precision_is_asked_for_only_where_it_helps(monkeypatch):
    """Half precision on a processor is slower, not faster.

    There is no hardware for it there, and mdcx is meant to run without CUDA,
    so a global dtype would trade a fivefold gain on one machine for a collapse
    on another.
    """
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert semantic._accelerator_kwargs() == {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pedido = semantic._accelerator_kwargs()
    assert pedido["torch_dtype"] is torch.float16
    assert pedido["attn_implementation"] == "sdpa"


def test_a_missing_torch_leaves_the_defaults(monkeypatch):
    """Asking the question must not be what breaks the install."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert semantic._accelerator_kwargs() == {}


def test_a_batch_is_filled_to_a_budget_not_to_a_count():
    """One long text must not set the width for a thousand short ones.

    The cost of a batch is rows times longest column, so the guarantee is on
    that product, not on how many texts got in.
    """
    textos = ["x" * 10] * 500 + ["y" * 2000]
    orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    lotes = list(semantic._batches(orden, textos, max_rows=1024, budget=24_000))

    assert [i for lote in lotes for i in lote] == orden, "no se pierde ningun texto"
    for lote in lotes:
        ancho = max(len(textos[i]) for i in lote)
        assert len(lote) * ancho <= 24_000 or len(lote) == 1, (
            f"un lote de {len(lote)} por {ancho} excede el presupuesto")


def test_a_single_text_longer_than_the_budget_still_goes():
    """A passage nothing can be grouped with is still encoded, alone."""
    textos = ["z" * 90_000]
    lotes = list(semantic._batches([0], textos, max_rows=1024, budget=24_000))
    assert lotes == [[0]]


def test_the_row_ceiling_holds():
    """A corpus of one-line headings must not assemble one enormous batch."""
    textos = ["x" * 5] * 5000
    orden = list(range(len(textos)))
    for lote in semantic._batches(orden, textos, max_rows=64, budget=10**9):
        assert len(lote) <= 64


@necesita_modelo
def test_the_vectors_come_back_in_the_order_the_texts_went_in():
    """The sorting is an implementation detail and must not escape.

    Encoded alone, a text has no order to lose; encoded in a batch it does. If
    the two disagree, every passage in a package carries another one's meaning
    and nothing reports it.
    """
    import numpy as np

    textos = ["a heading",
              "b " * 400,
              "c short",
              "d " * 90,
              "e a passage of middling length " * 8]
    juntos = semantic.encode(textos)
    solos = np.vstack([semantic.encode([t]) for t in textos])

    propio = (juntos * solos).sum(axis=1)
    assert propio.min() > 0.99, f"algun vector no es el de su texto: {propio}"


@necesita_modelo
def test_the_vectors_are_single_precision_whatever_produced_them():
    """Reduced precision is how the work is done, not what the package stores."""
    import numpy as np

    v = semantic.encode(["one", "two"])
    assert v.dtype == np.float32


@necesita_modelo
def test_nothing_to_encode_is_not_a_failure():
    """An empty call returns an empty result of the right width."""
    v = semantic.encode([])
    assert v.shape == (0, semantic.dimensions())
