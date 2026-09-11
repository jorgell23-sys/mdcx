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

"""Ordering an answer assembled from several packages.

Reported by a consumer serving 66 packages: a query whose material was in the
harvest returned its first four places to textbooks that did not contain the
words. Seven packages that declared every term known -- the ones holding the
subject -- appeared nowhere.

Their diagnosis was that BM25 scores from different packages are on different
scales, which is true and is fixed here. It was not what produced that ordering.
The merge was by reciprocal rank, which uses position and not score, and the
keys carry the package name, so no item appears in two lists: every list
contributes 1/(k+1) to its own first place, they all tie, and a stable sort
leaves the order the packages were listed in. The reply was ordered by the
environment variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx import archive  # noqa: E402
from mdcx.semantic import fuse  # noqa: E402

QUERY = "Carmichael numbers Korselt criterion squarefree"

ANSWER = ("Korselt criterion: a composite number is a Carmichael number if and "
          "only if it is squarefree and p-1 divides n-1 for all prime divisors p.")


def _package(tmp_path, name, documents):
    folder = tmp_path / name
    folder.mkdir()
    for title, text in documents.items():
        (folder / f"{title}.md").write_text(f"# {title}\n\n{text}\n",
                                            encoding="utf-8")
    target = tmp_path / f"{name}.mdcx"
    archive.pack(folder, target, "k")
    return archive.open_package(target, "k")[0]


def _two(tmp_path):
    """A small package of unrelated prose, and a large one holding the answer.

    The small one first, which is how the consumer's were configured: their own
    collections before the sixty of harvest.
    """
    textbook = _package(tmp_path, "textbook", {
        "notation": "Exponents and scientific notation for numbers in astronomy.",
        "surveys": "Public opinion surveys and statistical methodologies for numbers."})
    harvest = _package(tmp_path, "harvest", {
        **{f"paper{i}": f"A paper about algebraic structures and numbers, number {i}."
           for i in range(40)},
        "carmichael": ANSWER})
    return textbook, harvest


# --- What produced the ordering ------------------------------------------------


def test_merging_by_rank_orders_by_the_order_of_the_packages():
    """The defect, in the abstract and without a corpus: with keys that no two
    lists share, reciprocal rank degenerates into the order of the lists.

    This is what made the smallest packages appear to win -- they correlate
    with nothing but their position in the configuration.
    """
    lists = [[f"package{p}-doc{r}" for r in range(3)] for p in range(66)]

    merged = fuse(lists)

    assert merged[:4] == ["package0-doc0", "package1-doc0",
                          "package2-doc0", "package3-doc0"], (
        "reciprocal rank no longer degenerates, so this test is measuring "
        "something else")


# --- One scale for all of them --------------------------------------------------


def test_the_package_holding_the_answer_wins_though_it_is_listed_last(tmp_path):
    """The retirement condition, in miniature: the work that states the
    criterion comes first, and the textbooks that merely share a common word
    do not."""
    textbook, harvest = _two(tmp_path)
    terms = archive.terms_of(QUERY)
    corpus = archive.corpus_statistics_over([textbook, harvest], terms)

    everything = []
    for name, connection in (("textbook", textbook), ("harvest", harvest)):
        everything.extend(
            dict(item, package=name)
            for item in archive.query(connection, QUERY, limit=4, corpus=corpus))
    everything.sort(key=lambda item: -float(item.get("score") or 0.0))

    assert everything[0]["document"] == "carmichael"
    assert everything[0]["package"] == "harvest"


def test_without_common_statistics_it_does_not(tmp_path):
    """The control: the same corpus and the same query, merged the old way.
    Without this the test above would pass for any reason at all."""
    textbook, harvest = _two(tmp_path)

    by_key, lists = {}, []
    for name, connection in (("textbook", textbook), ("harvest", harvest)):
        items = []
        for item in archive.query(connection, QUERY, limit=4):
            key = (name, item["document"], item["passage"][:120])
            by_key[key] = dict(item, package=name)
            items.append(key)
        lists.append(items)

    merged = [by_key[k]["document"] for k in fuse(lists)[:4]]

    assert merged[0] != "carmichael", (
        "the old path already ranked it first, so the fix is unmeasured here")


# --- The statistics themselves --------------------------------------------------


def test_the_figures_are_the_sum_of_the_packages(tmp_path):
    """A term's rarity is a property of the corpus the score is computed over,
    so serving several packages means one corpus made of all of them."""
    textbook, harvest = _two(tmp_path)

    alone_t = archive._corpus_scale(textbook)[0]
    alone_h = archive._corpus_scale(harvest)[0]
    passages, _mean, frequencies = archive.corpus_statistics_over(
        [textbook, harvest], ["numbers"])

    assert passages == alone_t + alone_h
    assert frequencies["numbers"] == (
        archive._frequencies_for(textbook, ["numbers"]).get("numbers", 0)
        + archive._frequencies_for(harvest, ["numbers"]).get("numbers", 0))


def test_the_mean_length_is_weighted_by_how_much_each_package_holds(tmp_path):
    """An unweighted average would let a package of eleven documents count as
    much as one of five thousand, which is the same error in another place."""
    textbook, harvest = _two(tmp_path)

    _n, mean, _df = archive.corpus_statistics_over([textbook, harvest], ["numbers"])
    small_n, small_mean = archive._corpus_scale(textbook)
    big_n, big_mean = archive._corpus_scale(harvest)

    expected = (small_mean * small_n + big_mean * big_n) / (small_n + big_n)
    assert abs(mean - expected) < 1e-9


def test_a_single_package_is_unaffected(tmp_path):
    """Nothing changes for the ordinary case: one package is already one
    scale, and it is not asked to gather anything."""
    _textbook, harvest = _two(tmp_path)

    with_own = archive.query(harvest, QUERY, limit=3)
    corpus = archive.corpus_statistics_over([harvest], archive.terms_of(QUERY))
    with_gathered = archive.query(harvest, QUERY, limit=3, corpus=corpus)

    assert [x["document"] for x in with_own] == [x["document"] for x in with_gathered]


def test_the_terms_are_the_ones_the_query_will_be_weighed_by():
    """Gathered before any package is asked, so the rule has to be reachable
    from outside -- reproducing it in the caller would be a second definition."""
    assert archive.terms_of(QUERY) == ["carmichael", "numbers", "korselt",
                                       "criterion", "squarefree"]
