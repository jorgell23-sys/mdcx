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

"""Transcribing a formula, which is not the same as extracting one.

The text layer of a PDF does not carry the structure of a formula: the bar of a
fraction is a drawn stroke and a superscript is loose text on another line. A
faithful extraction of that layer therefore gives a quotient as one line, and
nothing in the pipeline is wrong -- the structure was never there to extract.

Measured by a consumer over 92 arXiv papers against their LaTeX source: a
formula written by a person matches the converted text in 0 to 7.6% of cases,
and the source in 98 to 100%. For the ~44,000 works they hold with no LaTeX
source there is nothing else to recover it from.

Recovering it means reading the image of the formula, which is recognition. It
is offered rather than assumed: it costs a model and time, it buys nothing on
prose, and what it produces is a transcription -- a different kind of claim
about the document, which can be wrong in ways extraction cannot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdcx.convert import engines  # noqa: E402


# --- Asked for, never assumed --------------------------------------------------


def test_formulas_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("MDCX_FORMULAS", raising=False)
    assert engines.formulas_wanted() is False


def test_asking_in_either_language(monkeypatch):
    """The switch travels to the workers through the environment, and a person
    setting it by hand writes what comes to mind."""
    for value in ("1", "true", "yes", "on", "si"):
        monkeypatch.setenv("MDCX_FORMULAS", value)
        assert engines.formulas_wanted() is True, value
    monkeypatch.setenv("MDCX_FORMULAS", "0")
    assert engines.formulas_wanted() is False


def test_the_converter_cache_tells_the_two_apart():
    """A cache key that omitted this would hand back a converter built for
    another question -- the defect would be a document silently converted
    without the recognition that was asked for."""
    source = Path(engines.__file__).read_text(encoding="utf-8")

    assert 'key = f"ocr={ocr},formulas={formulas}"' in source


# --- Said before the run, not during it ----------------------------------------


def test_a_missing_model_is_reported_with_the_command_to_get_it(monkeypatch,
                                                                tmp_path):
    """The converter builds without the model and fails on the first page that
    holds a formula, once per document, from inside docling. And mdcx pins the
    artifacts folder and works offline when it has local models, so nothing
    fetches it on the way past."""
    monkeypatch.setattr(engines, "local_artifacts_path", lambda: tmp_path)

    missing = engines.formula_model_missing()

    assert missing is not None
    assert engines.FORMULA_MODEL in missing
    assert "docling-tools models download-hf-repo" in missing


def test_a_present_model_is_not_complained_about(monkeypatch, tmp_path):
    (tmp_path / engines.FORMULA_MODEL.replace("/", "--")).mkdir(parents=True)
    monkeypatch.setattr(engines, "local_artifacts_path", lambda: tmp_path)

    assert engines.formula_model_missing() is None


def test_without_a_pinned_folder_docling_fetches_what_it_needs(monkeypatch):
    """Nothing to check: the automatic download is available, which is the
    case this must not refuse."""
    monkeypatch.setattr(engines, "local_artifacts_path", lambda: None)

    assert engines.formula_model_missing() is None


def test_the_run_refuses_rather_than_failing_per_document():
    """Settled before the corpus is touched."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "cli.py").read_text(encoding="utf-8")

    assert "Cannot transcribe formulas:" in source
    assert "engines.formula_model_missing()" in source


def test_asking_for_formulas_without_the_engine_says_so():
    """The recogniser lives inside the structured engine. The consumer who
    asked for this converts with --no-docling, because the cheap path is many
    times faster on their material, and would have got identical output with no
    indication why."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mdcx" / "cli.py").read_text(encoding="utf-8")

    assert "--formulas needs the structured engine" in source



# --- Neither engine answers both questions -------------------------------------


def _fake_reading(monkeypatch, plain, rich):
    """Two engines: one that covers the text, one that transcribes formulas."""
    from mdcx.convert import convert as convert_module

    def engines_for(job, ref_meta, use_docling):
        return [("nativo", lambda p: (plain, {}, None)),
                ("docling", lambda p: (rich, {}, None))]

    monkeypatch.setattr(convert_module, "_candidates", engines_for)
    monkeypatch.setattr(convert_module.extract, "reference_text",
                        lambda path, kind: (plain, {"pages": 1}))


def _job(tmp_path, name="paper"):
    from mdcx.convert.paths import Job

    source = tmp_path / f"{name}.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    return Job(source=source, rel_source=Path(f"{name}.pdf"),
               rel_target=Path(f"{name}.md"), kind="pdf", size=1)


def test_the_transcription_is_offered_to_whatever_reading_wins(monkeypatch,
                                                               tmp_path):
    """Measured on a twenty-page paper: native extraction covered the text
    entirely and carried no formula, while the structured engine transcribed
    fifty-six and lost 11% of the words. Choosing between them gives up one of
    the two, and choosing the structured reading quietly loses text -- which is
    what the verification exists to prevent.

    So neither is given up: whichever reading wins keeps the text, and the
    transcription is offered to it. What is checked here is that the offer is
    made, and made with the structured engine's Markdown. Which reading the
    score prefers is a separate decision with its own tests, and this must not
    depend on it.
    """
    from mdcx.convert import convert as convert_module

    monkeypatch.setenv("MDCX_FORMULAS", "1")
    plain = "The lemma states the identity for every prime p and q.\n"
    rich = plain + "\nand $$z(u) = a/b$$ with $x^2+y^2=z^2$ besides\n"
    _fake_reading(monkeypatch, plain, rich)

    offered = {}
    real = convert_module._formula_appendix

    def watched(transcribed, chosen):
        offered["transcribed"] = transcribed
        return real(transcribed, chosen)

    monkeypatch.setattr(convert_module, "_formula_appendix", watched)
    convert_module.convert_one(_job(tmp_path), tmp_path / "out")

    assert "transcribed" in offered, "the transcription was never offered"
    assert "x^2+y^2=z^2" in offered["transcribed"], (
        "what was offered is not the structured engine's reading")


def test_a_document_without_the_switch_is_not_offered_one(monkeypatch, tmp_path):
    """Nothing is transcribed and nothing is appended unless it was asked for."""
    from mdcx.convert import convert as convert_module

    monkeypatch.delenv("MDCX_FORMULAS", raising=False)
    plain = "Ordinary prose with no mathematics in it at all.\n"
    _fake_reading(monkeypatch, plain, plain)

    record = convert_module.convert_one(_job(tmp_path, "note"), tmp_path / "out")

    assert "formulas" not in record
    written = (tmp_path / "out" / "note.md").read_text(encoding="utf-8")
    assert "## Formulas" not in written



def test_the_formulas_are_not_appended_twice(monkeypatch, tmp_path):
    """When the structured engine is itself the chosen reading, the formulas
    are already in the text."""
    from mdcx.convert import convert as convert_module

    rich = "Prose and $$z(u) = \frac{a}{b}$$ and $x^2+y^2=z^2$ and more.\n"

    assert convert_module._formula_appendix(rich, rich) == ("", 0)


def test_nothing_is_appended_when_no_formula_was_found(monkeypatch, tmp_path):
    """Prose gains nothing, and an empty heading is worse than no heading."""
    from mdcx.convert import convert as convert_module

    assert convert_module._formula_appendix("plain prose", "plain prose") == ("", 0)
