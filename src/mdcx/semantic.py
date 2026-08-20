"""Dense multilingual retrieval, so that a query reaches documents written in
another language.

Lexical retrieval matches words. A Spanish query cannot reach a German document
about the same subject, because the words are not there: retrieval by word is
retrieval within a language, however well the words are tokenised. Reaching
across languages requires representing meaning rather than spelling, which is
what a multilingual embedding model does. It maps a sentence and its translation
to nearby points, so that proximity in the vector space stands for sameness of
meaning across the languages the model was aligned on.

The two methods answer different questions and are kept side by side rather than
one replacing the other. The lexical index knows that a document contains a
term; the dense index knows that a document means something similar. Measured on
a corpus written in thirty-four languages, the lexical engine alone retrieves
4.4 per cent of the documents on a subject when the query is written in another
language, and the dense engine alone loses precision on the language of the
query itself. Fused, each covers what the other cannot.

This module is optional. Without it a package is queried lexically, exactly as
before, and nothing fails: the extra is what a corpus in several languages
needs, not what every corpus needs.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# The default model is chosen by measurement rather than by benchmark rank, and
# what decides is the language the model handles worst rather than its average: a
# model that averages well while collapsing on one language does not serve
# whoever reads in that language.
#
# Measured on FLORES-200, where the task is to find a sentence given its
# translation, among candidates drawn from the same corpus:
#
#   BGE-M3        100.0% mean, 99.8% on its worst language, 98.0% on its worst pair
#   LaBSE          99.5% mean, 97.7% worst language, 96.0% worst pair
#   e5-large       98.9% mean, 96.5% worst language, 92.0% worst pair
#   e5-small       97.0% mean, 94.4% worst language, 92.0% worst pair
#   granite-97m    95.4% mean, 91.3% worst language, 84.0% worst pair
#
# BGE-M3 leads on all three, and on the third by the widest margin, which is the
# one that decides.
DEFAULT_MODEL = "BAAI/bge-m3"

# Prefixes are part of a model's protocol rather than decoration. E5 was trained
# with a query and a passage carrying different prefixes, and omitting them
# measures a misuse of the model instead of the model.
MODEL_PREFIXES = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
}

_LOCK = threading.Lock()
_LOADED: dict[str, object] = {}


class MissingDependency(RuntimeError):
    """Raised when semantic retrieval is asked for without its dependencies."""


def model_name() -> str:
    """The model to use, which the environment may override."""
    return os.environ.get("MDCX_MODEL", DEFAULT_MODEL)


def available() -> bool:
    """Whether semantic retrieval can run in this interpreter."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load(name: str | None = None):
    """Return the encoder, loading it once per process.

    Loading takes seconds and holds hundreds of megabytes, so a server that
    answers many queries must not repeat it. The lock makes the first call safe
    when several threads arrive at once, which is how the MCP server behaves.
    """
    name = name or model_name()
    with _LOCK:
        if name in _LOADED:
            return _LOADED[name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise MissingDependency(
                "Semantic retrieval needs the multilingual extra: "
                "pip install 'mdcx[multilingual]'") from e
        # The device is left to the library, which selects the accelerator when
        # one is present and the processor otherwise.
        model = SentenceTransformer(name, trust_remote_code=True)
        _LOADED[name] = model
        return model


def prefixes(name: str | None = None) -> tuple[str, str]:
    """The query and passage prefixes of a model, empty when it uses none."""
    return MODEL_PREFIXES.get(name or model_name(), ("", ""))


def encode(texts: list[str], role: str = "passage", name: str | None = None,
           batch_size: int = 32):
    """Encode texts as unit vectors.

    The role selects the prefix. A text encoded as a passage and the same text
    encoded as a query are different vectors, by design of the models that use
    prefixes.
    """
    if role not in ("query", "passage"):
        raise ValueError(f"role must be query or passage, not {role!r}")
    model = load(name)
    query_prefix, passage_prefix = prefixes(name)
    prefix = query_prefix if role == "query" else passage_prefix
    return model.encode([prefix + t for t in texts],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=batch_size)


def dimensions(name: str | None = None) -> int:
    """The length of the vectors a model produces."""
    return int(load(name).get_sentence_embedding_dimension())


def fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked lists by reciprocal rank.

    Each list contributes 1/(k+position) to the items it ranks. Position is used
    rather than score because the two engines score on scales that have no
    common meaning: a BM25 score of 8 and a cosine similarity of 0.8 cannot be
    added, and normalising them introduces a weighting that nothing justifies.
    Rank is the one thing both engines agree on.

    The constant damps the influence of the first places, so that an item ranked
    highly by both engines outranks an item ranked first by one and ignored by
    the other.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + position)
    return sorted(score, key=lambda item: -score[item])
