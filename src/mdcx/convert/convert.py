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

"""Conversion of a single document, with fidelity verification.

Several engines are attempted and the best result is selected by a composite
score. Content present in the original but absent from the Markdown is
appended verbatim in a recovery section rather than reported as lost.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import chapters, compact as compact_module, engines, extract, verify
from .. import console
from .paths import Job, file_digest, to_pseudopath

_H_RE = re.compile(r"(?m)^#{1,6}\s+\S")
_TBL_RE = re.compile(r"(?m)^\s*\|[-: |]+\|\s*$")
_LI_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")

PLAN_MAX_PAGES = 2

RECOVERY_LINE_RATIO = 0.5

def _yaml_escape(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

def _front_matter(job: Job, record: dict) -> str:
    v = record["verification"]
    lines = [
        "---",
        f"title: {_yaml_escape(job.rel_source.stem)}",
        f"source: {_yaml_escape(job.rel_source.name)}",
        f"source_pseudopath: {_yaml_escape(job.source_pseudopath)}",
        f"markdown_pseudopath: {_yaml_escape(job.pseudopath)}",
        f"folder: {_yaml_escape(job.rel_source.parent.as_posix() or '.')}",
        f"source_format: {job.kind}",
        f"bytes_source: {job.size}",
        f"sha256_source: {record['digest']}",
        f"engine: {record['engine']}",
        f"ocr: {str(record.get('ocr', False)).lower()}",
        f"converted_utc: {record['converted_at']}",
        f"fidelity: {v.get('coverage') if v.get('coverage') is not None else 'not-measurable'}",
        f"numeric_fidelity: {v.get('numeric_coverage') if v.get('numeric_coverage') is not None else 'not-measurable'}",
        f"verification_status: {v.get('status')}",
    ]
    if record.get("pages"):
        lines.append(f"pages: {record['pages']}")
    # A sample says so, and says what it is a sample of. Without both numbers a
    # sample of twenty pages of a book of six hundred is, by construction, a
    # truncated document that looks whole.
    if record.get("sampled_from"):
        lines.append(f"pages_total: {record['sampled_from']}")
        lines.append("sampled: true")
    if record.get("recovered_lines"):
        lines.append(f"recovered_lines: {record['recovered_lines']}")
    lines.append("---")
    return "\n".join(lines)

def _recovery_block(reference: str, markdown: str) -> tuple[str, int]:
    """Build the recovery section with original lines missing from the Markdown."""
    ref_tokens = verify.tokenize(reference)
    md_tokens = verify.tokenize(verify.strip_markdown_noise(markdown))
    deficit = ref_tokens - md_tokens  # Counter subtraction discards non-positive counts
    if not deficit:
        return "", 0

    remaining = sum(deficit.values())
    recovered: list[str] = []
    for raw_line in reference.splitlines():
        if remaining <= 0:
            break
        line = raw_line.strip()
        if not line:
            continue
        tokens = verify.tokenize(line)
        total = sum(tokens.values())
        if not total:
            continue
        hits = sum(min(c, deficit.get(t, 0)) for t, c in tokens.items())
        if hits / total < RECOVERY_LINE_RATIO:
            continue
        recovered.append(line)
        for t, c in tokens.items():
            pending = deficit.get(t, 0)
            if not pending:
                continue
            used = c if c < pending else pending
            if used == pending:
                del deficit[t]
            else:
                deficit[t] = pending - used
            remaining -= used

    fragments = sorted(deficit.elements()) if deficit else []

    if not recovered and not fragments:
        return "", 0

    parts = [
        "\n\n---\n\n",
        "## Recovery appendix\n\n",
        "> Content present in the source document that the structured conversion "
        "engine did not include. Appended verbatim, without reformatting, so that "
        "the Markdown loses no information relative to the source.\n\n",
    ]
    if recovered:
        parts.append("```text\n" + "\n".join(recovered) + "\n```\n")
    if fragments:
        parts.append(
            "\n**Isolated fragments with no line of their own:**\n\n"
            "```text\n" + " ".join(fragments) + "\n```\n"
        )
    return "".join(parts), len(recovered)

def _is_drawing(job: Job, ref_meta: dict) -> bool:
    """Report whether the document behaves like a drawing rather than text."""
    if job.kind != "pdf":
        return False
    pages = ref_meta.get("pages") or 0
    if not 0 < pages <= PLAN_MAX_PAGES:
        return False
    if ref_meta.get("embedded_images", 0) < 1:
        return False
    return True

def _candidates(job: Job, ref_meta: dict, use_docling: bool) -> list[tuple[str, callable]]:
    """Choose which engines to try, and in what order, by document type.

    Reading a page with a vision model costs about a second and extracting its
    text costs three milliseconds, so which engine is tried first decides what
    converting a library costs. The cheap one goes first and the expensive one
    is kept for the documents it actually helps -- the ones whose tables the
    cheap engine could not find, or whose pages carry no text layer at all.

    A document that needs OCR has nothing to extract, so there the order does
    not apply: only the model can read it.
    """
    native = engines.NATIVE.get(job.kind)
    needs_ocr = bool(ref_meta.get("needs_ocr"))

    if job.kind == "text":
        return [("nativo", lambda p: engines.native_text(p))]

    out: list[tuple[str, callable]] = []
    docling_ok = use_docling and engines.docling_available()

    if docling_ok and needs_ocr:
        out.append(("docling-ocr", lambda p: engines.docling_convert(p, ocr=True)))

    if _is_drawing(job, ref_meta) and not needs_ocr:
        if native:
            out.append(("nativo", native))
        return out

    # A chapter is converted from pages cut out of the book, and cutting them
    # leaves the bookmarks behind. The headings have to be fetched from the book
    # itself, or the chapter arrives as a wall of prose with no section titles
    # -- which are among the best answers a search can return, and which the
    # author already wrote. Both cheap engines need them: giving them to one and
    # not the other leaves the second producing nothing structural, being
    # rejected for it, and sending the document to layout analysis anyway.
    titles = None
    if job.kind == "pdf" and (job.is_chapter or job.sample_pages):
        try:
            from . import pdf as _pdf

            if job.sample_pages:
                # A sample loses its bookmarks the same way a chapter does, and
                # a sample without headings is a wall of prose -- which is the
                # opposite of what a sample is for, since the section titles
                # are most of what says whether the book is worth converting.
                titles = _pdf.outline_for_pages(job.source, list(job.sample_pages))
            elif job.page_range:
                titles = _pdf.outline_for_range(job.source, *job.page_range)
        except Exception:  # noqa: BLE001
            titles = None

    if native and job.kind == "pdf":
        out.append(("nativo", lambda p, t=titles: engines.native_pdf(p, t)))
    elif native:
        out.append(("nativo", native))
    if docling_ok and job.kind == "pdf":
        # Between the two extremes: everything extracted cheaply, and only the
        # pages that announce a table the cheap path could not produce read by
        # the model. A book with tables on a tenth of its pages then pays for
        # layout analysis on a tenth of its pages.
        # The same titles the cheap path gets. Without them this engine returns
        # a chapter with no structural marks, is rejected for having none, and
        # the document goes to layout analysis after all -- which is the whole
        # of what fetching the headings was for.
        out.append(("hibrido", lambda p, t=titles: engines.hybrid_pdf(p, t)))
    if docling_ok:
        out.append(("docling", lambda p: engines.docling_convert(p, ocr=False)))
    return out

MIN_USABLE_COVERAGE = 0.60

MIN_TOKENS_WITHOUT_REFERENCE = 15

def _score(md: str, v: dict) -> tuple:
    """Composite score used to choose among engine outputs."""
    cov = v.get("coverage")
    usable = 1 if (cov is None and md.strip()) or (cov is not None and cov >= MIN_USABLE_COVERAGE) else 0
    structure = len(_H_RE.findall(md)) * 2 + len(_TBL_RE.findall(md)) * 5 + len(_LI_RE.findall(md))
    return (usable, structure, cov or 0.0, len(md))

# How much less of the document a later engine may cover and still be preferred.
# Half a hundredth absorbs the difference between two correct readings of the
# same page; beyond that the text is being dropped rather than rearranged. A
# whole hundredth is too generous: the reported cases include one at exactly
# 0.990 against 1.000, which would have sat on the boundary and been kept, and
# one per cent of a chapter of two hundred thousand characters is two thousand
# characters of text.
COVERAGE_TOLERANCE = 0.005


def _covers_less(v: dict, attempts: list[dict]) -> bool:
    """Whether this result reads less of the document than one already tried.

    No engine should be preferred to one that already covered more. The
    composite score ranks structure above coverage, so a result that recovers a
    table while dropping the prose around it wins on points and publishes less
    text than the pipeline already had in hand. Measured over 214 chapters that
    happened in 45 of them, and in one the chosen result was 16,624 characters
    shorter than the one already extracted.

    It applies to every engine and not only to the cheap ones: layout analysis
    returns more characters and covers fewer of the words that are actually in
    the document, so it wins on length while losing text.
    """
    covered = v.get("coverage")
    if covered is None:
        return False        # nothing measured here to compare against
    best = max((a.get("coverage") or 0.0) for a in attempts) if attempts else 0.0
    return covered < best - COVERAGE_TOLERANCE


# The coverage above which there is no text left for another engine to recover.
# Not 1.0: two correct readings of the same page differ in the last decimals,
# and holding out for exactness would run the expensive engine over documents
# that were already read whole.
COVERAGE_COMPLETE = 0.995


def _nothing_left_to_recover(attempts: list[dict]) -> bool:
    """Whether some engine has already read the whole document.

    Layout analysis is worth its cost when text is missing. When the cheap path
    has already covered the document end to end, what it could still add is
    structure -- and it charges about a hundred times more per page for it.

    Measured on chapters of a textbook: the native engine covered 1.0000 and the
    hybrid 0.9924, which counts as a warning rather than a pass, so the search
    continued and layout analysis ran over a document with nothing left to find.
    Those chapters took twenty seconds where they now take two.
    """
    return any((a.get("coverage") or 0.0) >= COVERAGE_COMPLETE for a in attempts)


def _good_enough(name: str, meta: dict, v: dict, score: tuple,
                 attempts: list[dict] | None = None) -> bool:
    """Whether this result makes running the next engine pointless.

    Stopping early is what the order of the engines is worth: putting the cheap
    one first saves nothing if the expensive one runs anyway. But stopping on
    the cheap engine needs more than a good score, because the thing it can
    quietly miss is a table, and a page whose table was read as running text
    still scores as covered -- the words are all there, in the wrong shape.

    So it is asked what it left unsettled: the pages that either announce a
    table or are ruled like one, and yielded none. An earlier version asked
    instead how many of the announced tables it had found, which counted labels
    rather than tables -- a book that labels a screenshot of a spreadsheet
    "Figure 4.2" announced nothing, so nothing was missing, so the whole
    document went unexamined. Three books in six were skipped that way.
    """
    # The hybrid engine is the end of the cheap path: it has already sent the
    # model every page the rest could not settle. If by then the document has
    # been read end to end, layout analysis has no text left to recover, and
    # what it would still add is structure at a hundred times the cost per page.
    # This is checked before the status, because a hybrid result sits at "warn"
    # for losing a thousandth of coverage against a native reading that is
    # already complete -- and that alone was sending whole chapters to the
    # expensive engine for nothing.
    # Structure is still required: a chapter read whole but returned as a wall
    # of prose is what sending it to layout analysis is for, and accepting it
    # here would undo that. What this allows is the opposite case -- a document
    # already read end to end, with its headings, that the hybrid engine left at
    # "warn" for trading a thousandth of coverage for a table.
    if (name == "hibrido" and score[0] == 1 and score[1] > 0
            and _nothing_left_to_recover(
                list(attempts or []) + [{"coverage": v.get("coverage")}])):
        return True

    if v.get("status") != "ok" or score[0] != 1 or score[1] <= 0:
        return False
    if name.startswith("docling"):
        return True
    if name == "hibrido":
        return True
    pending = meta.get("unresolved")
    if pending is None:
        return False        # an engine that cannot answer defers to the model
    return not pending

def convert_one(job: Job, output_root: Path, use_docling: bool = True,
                save_lossless: bool = True, compact: bool = True) -> dict:
    """Convert one file and write its mirrored .md. Returns the index record."""
    started = time.time()
    record: dict = {
        "source_pseudopath": job.source_pseudopath,
        "markdown_pseudopath": job.pseudopath,
        "source_name": job.rel_source.name,
        "folder": job.rel_source.parent.as_posix() or ".",
        "format": job.kind,
        "bytes": job.size,
        "converted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": "none",
        "errors": [],
    }

    try:
        record["digest"] = file_digest(job.source)
    except Exception as exc:  # noqa: BLE001
        record["digest"] = ""
        record["errors"].append(f"hash: {exc}")

    source = job.source
    temporary: Path | None = None
    if job.is_chapter:
        try:
            temporary = Path(tempfile.gettempdir()) / (
                f"pdftomd_{os.getpid()}_{job.chapter_index:03d}_{job.source.stem[:40]}.pdf"
            )
            chapters.extract_range(job.source, job.page_range[0], job.page_range[1], temporary)
            source = temporary
        except Exception as exc:  # noqa: BLE001
            record["errors"].append(f"page extract: {type(exc).__name__}: {exc}")
            temporary = None
    elif job.sample_pages:
        try:
            from . import pdf as _pdf

            temporary = Path(tempfile.gettempdir()) / (
                f"mdcx_sample_{os.getpid()}_{job.source.stem[:40]}.pdf")
            # Gathered into one document rather than converted page by page: an
            # engine that charges per document would otherwise pay that charge
            # once per page of the sample.
            _pdf.extract_page_list(job.source,
                                   [p - 1 for p in job.sample_pages], temporary)
            source = temporary
        except Exception as exc:  # noqa: BLE001
            record["errors"].append(f"sample extract: {type(exc).__name__}: {exc}")
            temporary = None

    if job.sample_pages:
        record["sampled_from"] = job.pages_total or 0
        record["sample_pages"] = list(job.sample_pages)

    record["chapter"] = {
        "title": job.chapter_title,
        "outline": job.chapter_index,
        "pages": list(job.page_range),
        "document_pseudopath": to_pseudopath(job.parent_target) if job.parent_target else None,
    } if job.is_chapter else None

    reference, ref_meta = extract.reference_text(source, job.kind)
    record["reference_meta"] = ref_meta
    if ref_meta.get("pages"):
        record["pages"] = ref_meta["pages"]

    best_md = ""
    best_v: dict | None = None
    best_engine = "none"
    best_score: tuple | None = None
    best_lossless = None
    best_ocr = False
    attempts: list[dict] = []

    for name, fn in _candidates(job, ref_meta, use_docling):
        try:
            md, meta, lossless = fn(source)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"engine": name, "error": f"{type(exc).__name__}: {exc}"})
            record["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        v = verify.compare(reference, md)
        score = _score(md, v)
        attempts.append({
            "engine": name,
            "coverage": v.get("coverage"),
            "status": v.get("status"),
            "chars": len(md),
            "score": list(score),
        })

        steps_back = _covers_less(v, attempts)

        if not steps_back and (best_score is None or score > best_score):
            best_md, best_v, best_engine, best_score = md, v, name, score
            best_lossless = lossless
            best_ocr = bool(meta.get("ocr"))
            if meta.get("pages"):
                record.setdefault("pages", meta["pages"])

        # Whether to stop does not depend on this result being the best one. A
        # result that reads less than an earlier attempt is not used -- that is
        # what `retrocede` decides above -- but the search can still end here if
        # what is already in hand is enough. Tying the two together kept sending
        # documents to layout analysis precisely when the cheap path had already
        # read them whole and the hybrid had merely traded a thousandth of
        # coverage for a table.
        if _good_enough(name, meta, v, score, attempts):
            break

    record["attempts"] = attempts
    record["engine"] = best_engine
    record["ocr"] = best_ocr

    # An engine that crashed is not the same as an engine that was not needed,
    # and until now both looked alike in the report. On Linux with CUDA every
    # docling attempt failed and the run still reported success.
    failed = [a["engine"] for a in attempts if a.get("error")]
    if failed:
        record["failed_engines"] = failed

    # Coverage is measured in words, and a native engine recovers every word of a
    # table while flattening it into a paragraph. Counting table rows exposes a
    # loss of structure that coverage alone reports as perfect.
    record["table_rows"] = sum(
        1 for line in best_md.splitlines() if line.lstrip().startswith("|"))

    if best_v is None:
        record["verification"] = {"status": "error", "measurable": False,
                                 "coverage": None, "numeric_coverage": None}
        record["ok"] = False
        record["seconds"] = round(time.time() - started, 2)
        return record

    recovered_lines = 0
    if best_v.get("measurable") and best_v.get("missing_tokens"):
        block, recovered_lines = _recovery_block(reference, best_md)
        if block:
            best_md = best_md.rstrip() + block
            best_v = verify.compare(reference, best_md)
            best_v["recovered"] = True

    record["recovered_lines"] = recovered_lines
    record["verification"] = best_v
    # A document is conforming when it was measured against its original and
    # met it. A document that exposes no text was never measured, so it is not
    # conforming, and it is not a failure either: the two are counted apart in
    # the index.
    record["ok"] = best_v.get("status") == "ok"

    if best_v.get("status") == "no-reference":
        content = len(verify.tokenize(best_md))
        if content < MIN_TOKENS_WITHOUT_REFERENCE:
            record["verification"]["status"] = "unreadable"
            record["verification"]["note"] = (
                "the original exposes no text and optical recognition produced no "
                "content: the document requires manual review"
            )
        else:
            record["verification"]["status"] = "only-ocr"
            record["verification"]["note"] = (
                "content comes from optical recognition only; there is no original "
                "text to verify it against. Visual review required."
            )

    target = output_root / job.rel_target
    target.parent.mkdir(parents=True, exist_ok=True)
    if job.is_chapter:
        header = (
            f"# {job.chapter_title}\n\n"
            f"> Capitulo {job.chapter_index} de *{job.rel_source.stem}* — "
            f"pages {job.page_range[0]} to {job.page_range[1]} of the original.  \n"
            f"> Full document: `{to_pseudopath(job.parent_target)}`\n\n"
        )
    else:
        header = f"# {job.rel_source.stem}\n\n"
    body = _front_matter(job, record) + "\n\n" + header + best_md.strip() + "\n"
    if compact:
        body, applied = compact_module.compact(body)
        record["compacted"] = applied
    target.write_text(body, encoding="utf-8")
    record["md_bytes"] = len(body.encode("utf-8"))

    if save_lossless and best_lossless is not None:
        lossless_path = output_root / "_lossless" / job.rel_target.with_suffix(".docling.json")
        lossless_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lossless_path.write_text(
                json.dumps(best_lossless, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            record["lossless_pseudopath"] = "@/_lossless/" + job.rel_target.with_suffix(".docling.json").as_posix()
        except Exception as exc:  # noqa: BLE001
            record["errors"].append(f"lossless: {exc}")

    if temporary is not None:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass

    record["seconds"] = round(time.time() - started, 2)
    return record

class DocumentTimeout(Exception):
    """A document took longer than the run allows for one."""


@contextlib.contextmanager
def _time_limit(seconds: float | None):
    """Give up on the body after so long, from inside the worker.

    A document that never finishes holds its worker for as long as the run
    lasts, and with six workers six such documents stop the conversion
    altogether. Measured on real material: two workers spent three hours and
    twenty minutes of processor time on documents of two to eight A4 pages,
    while equivalent ones in the same batch took six seconds natively and
    thirty-five through layout analysis. The content is ordinary; something in
    the structured engine does not terminate on it.

    Raised into the working thread rather than by killing the process, so the
    worker records the document as unfinished and goes on to the next one. A
    killed worker would take the pool with it and lose the batch, which is the
    outcome this exists to avoid.

    The interruption lands when the interpreter next runs bytecode, so a call
    that stays inside a C extension is not cut short at the instant the limit
    expires -- the model returns to Python between layers, which is where it
    takes effect. It is a bound on the document, not on the instruction.
    """
    if not seconds or seconds <= 0:
        yield
        return

    import ctypes
    import threading

    target = threading.get_ident()
    fired = threading.Event()

    def expire():
        fired.set()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(target), ctypes.py_object(DocumentTimeout))

    alarm = threading.Timer(seconds, expire)
    alarm.daemon = True
    alarm.start()
    try:
        yield
    except DocumentTimeout:
        raise DocumentTimeout(
            f"gave up after {seconds:.0f}s") from None
    finally:
        alarm.cancel()
        if fired.is_set():
            # The exception may have been raised while another one was already
            # travelling, in which case it is still pending against this thread
            # and would surface inside whatever runs next. Clear it.
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(target), ctypes.c_void_p(0))


def convert_one_safe(args) -> dict:
    """Process pool wrapper: one broken file must not bring down the batch."""
    job, output_root, use_docling, save_lossless, *rest = args
    compact = rest[0] if rest else True
    timeout = rest[1] if len(rest) > 1 else None
    # A worker is a new process and does not inherit the streams the parent
    # configured, so it configures its own before writing anything.
    console.configure()

    label = job.rel_source.name
    if job.is_chapter:
        label += f"  [cap. {job.chapter_index}: {job.chapter_title[:40]}]"
    console.safe_print(f">>> {label}", flush=True)
    try:
        with _time_limit(timeout):
            return convert_one(job, output_root, use_docling, save_lossless, compact)
    except DocumentTimeout as expired:
        console.safe_print(f"!!! {label}: {expired}", flush=True)
        return {
            "source_pseudopath": job.source_pseudopath,
            "markdown_pseudopath": job.pseudopath,
            "source_name": job.rel_source.name,
            "folder": job.rel_source.parent.as_posix() or ".",
            "format": job.kind,
            "bytes": job.size,
            "engine": "none",
            "ok": False,
            # Its own status rather than "error": nothing was found wrong with
            # this document, and a reader deciding what to retry needs to tell
            # "the conversion failed on it" from "it was still going".
            "verification": {"status": "timeout", "measurable": False,
                             "coverage": None, "numeric_coverage": None},
            "errors": [f"DocumentTimeout: {expired}"],
        }
    except Exception:  # noqa: BLE001
        return {
            "source_pseudopath": job.source_pseudopath,
            "markdown_pseudopath": job.pseudopath,
            "source_name": job.rel_source.name,
            "folder": job.rel_source.parent.as_posix() or ".",
            "format": job.kind,
            "bytes": job.size,
            "engine": "none",
            "ok": False,
            "verification": {"status": "error", "measurable": False,
                             "coverage": None, "numeric_coverage": None},
            "errors": [traceback.format_exc(limit=3)],
        }
    finally:
        # A permit for the card is returned by the block that took it, and that
        # block is not guaranteed to end: layout analysis can leave a thread in
        # a blocking call and abandon it, and the timeout can cut the document
        # short between taking the permit and entering the body. Either way the
        # permit would be gone for the rest of the run -- and with two of them,
        # two such documents were enough to leave every worker waiting on a
        # turn that never came.
        #
        # The document is the boundary that always ends, so it is where the
        # permits are squared up.
        stranded = engines.release_turns()
        if stranded:
            console.safe_print(
                f"!!! {label}: {stranded} turn(s) on the card recovered",
                flush=True)
