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

"""Command line interface for the converter.

Converts a folder of documents to Markdown, mirroring its directory
structure, verifying each file against its original and writing a global
index of the result.

    python -m mdcx.cli --input ./Documents --output ./Documents_md
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import console  # noqa: E402
from .convert import chapters, engines, index  # noqa: E402
from .convert import pdf as _pdf  # noqa: E402
from .convert import compact as compact_module  # noqa: E402
from .convert.convert import convert_one_safe  # noqa: E402
from .convert.paths import (  # noqa: E402
    LANE_CPU,
    LANE_GPU,
    estimated_cost,
    inspect_job,
    file_digest,
    make_chapter_jobs,
    plan_jobs,
    to_pseudopath,
)

ROOT = Path.cwd()
CERT_BUNDLE = ROOT / "tool" / "certs" / "ca-bundle.pem"

# What is left for whoever is using the machine, as a share of it rather than a
# fixed number of processors. One free core is a different promise on a machine
# of four than on one of thirty-two: on the first it is a quarter of the machine
# and on the second it is nothing, and it is the second where a converting
# library takes the desk away from its owner.
RESERVED_SHARE = 0.20

def _default_max_cores() -> int:
    """How many processes to run at once unless told otherwise.

    The whole of the machine less the reserve, and no fixed ceiling above that.
    A number that does not grow with the machine is the same defect as a reserve
    that does not: it was the bare constant 8, which asks for twice a machine of
    four and for a quarter of a machine of thirty-two, and a converter that
    cannot use a large machine is a converter that has to be told to, every
    time, by someone who first has to notice.

    That mattered little while almost every document queued in a lane of three
    whatever the cap said. Now that the lanes are filled, this is what decides
    the real concurrency.

    For the record rather than as a rule: on the machine this was developed on,
    twelve logical processors, eight was measured as comfortable and the share
    here allows nine. --max-cores is how a machine says something different.
    """
    import os as _os

    available = _os.cpu_count() or 4
    return max(1, int(available * (1.0 - RESERVED_SHARE)))


MAX_CORES_DEFAULT = _default_max_cores()

def _configure_tls() -> None:
    """Use the local CA bundle if present."""
    if CERT_BUNDLE.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(var, str(CERT_BUNDLE))

    if engines.local_artifacts_path() is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# How long one document may take before the run gives up on it and moves on.
#
# There is material the structured engine does not finish on. Measured: two
# workers spent three hours and twenty minutes of processor time on documents of
# two to eight A4 pages, while equivalent documents in the same batch took six
# seconds natively and up to thirty-five through layout analysis. Nothing about
# those documents is unusual -- compared against sixty that convert normally
# they have fewer pages, the same megabytes, the same images per page and
# slightly more text -- so there is nothing to screen for beforehand. Roughly
# 4% of a real collection behaved this way.
#
# Without a limit one such document holds its worker for the length of the run,
# and six of them stop the conversion altogether: the batch cannot close because
# it waits for everyone.
#
# Twenty minutes rather than something tight. A long book with many tables can
# legitimately take a while, and abandoning a good document costs more than
# waiting too long for a bad one -- the whole of the document's work is lost,
# while waiting costs one worker for the excess. It is a backstop, not a budget.
DOCUMENT_TIMEOUT_SECONDS = 1200.0

# After how many documents a worker is replaced by a fresh one.
#
# Giving up on a document does not undo what it started. Layout analysis says so
# itself -- "thread is likely stuck in a blocking call and will be abandoned" --
# and an abandoned thread keeps running: measured, three of them held a core
# each at 100% for twenty minutes while producing nothing, and the run crawled
# to a stop. A Python thread cannot be cancelled, so the only way to get the
# core back is to replace the process holding it.
#
# Every so many documents rather than on demand, because a pool cannot be told
# to retire one particular worker; what it offers is a lifetime. Fifty bounds
# the loss to fifty documents on one worker instead of to the rest of the run,
# and costs one reload of the models in that time -- against documents that take
# tens of seconds each, a small share.
RECYCLE_AFTER_DOCUMENTS = 50

PROGRESS_FILENAME = "_progress.jsonl"

# What the file was called before. A run interrupted under the old name is
# still picked up, so the work already done is not repeated.
LEGACY_PROGRESS_FILENAME = "_progreso.jsonl"

def _load_cache(output_root: Path) -> dict:
    """Previous work, indexed by the pseudopath of the target Markdown."""
    cache: dict = {}

    manifest = output_root / "_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for d in data.get("documents", []):
                if d.get("markdown_pseudopath"):
                    cache[d["markdown_pseudopath"]] = d
        except Exception:
            pass

    for filename in (LEGACY_PROGRESS_FILENAME, PROGRESS_FILENAME):
        progress = output_root / filename
        if not progress.exists():
            continue
        try:
            with progress.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("markdown_pseudopath"):
                        cache[d["markdown_pseudopath"]] = d
        except Exception:
            pass

    return cache

def _print_progress(done: int, total: int, lane: str, rec: dict) -> None:
    v = rec.get("verification") or {}
    cov = v.get("coverage")
    cov_s = f"{cov * 100:6.2f}%" if cov is not None else "  n/a "
    flag = "ok " if rec.get("ok") else "!! "
    console.safe_print(
        f"[{done:>3}/{total}] {lane.upper():<3} {flag} {cov_s} "
        f"{rec.get('engine', '?'):<11} {rec.get('seconds', 0):>6.1f}s "
        f"{rec['source_name'][:52]}",
        flush=True,
    )

def _error_record(job, exc: Exception) -> dict:
    return {
        "source_pseudopath": job.source_pseudopath,
        "markdown_pseudopath": job.pseudopath,
        "source_name": job.rel_source.name,
        "folder": job.rel_source.parent.as_posix() or ".",
        "format": job.source.suffix.lower().lstrip("."),
        "bytes": job.size,
        "engine": "none",
        "ok": False,
        "verification": {"status": "error", "measurable": False,
                         "coverage": None, "numeric_coverage": None},
        "errors": [f"{type(exc).__name__}: {exc}"],
    }


# What one worker holds on the card once its batch is full: the models, plus the
# images of a full batch. Measured on a 6 GB card as 206 MiB with the models
# alone, 364 after two pages and 1,384 once the batch of eight is complete. The
# peak is the number that matters; sizing on a sample taken before the batch
# fills says more workers fit than do, and the run then spends itself competing
# for memory rather than converting.
#
# This is a starting estimate and not a measurement taken here: measuring it
# properly means loading the models and inferring a full batch before any work
# begins, which costs more than the sizing saves. MDCX_VRAM_PER_PROCESS says
# otherwise for a card whose models measure differently.
VRAM_PER_WORKER_MIB = 1384

# The former spelling. It is still read so that a machine that already sets it
# keeps the sizing it was given.
VRAM_ENV_VARS = ("MDCX_VRAM_PER_PROCESS", "MDCX_VRAM_POR_PROCESO")


def _vram_per_worker_mib() -> int:
    for name in VRAM_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            return VRAM_PER_WORKER_MIB
    return VRAM_PER_WORKER_MIB


def _free_vram_mib() -> int | None:
    """Video memory available right now, or None where there is no card to ask."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:  # noqa: BLE001 - no card, no driver, no torch: same answer
        return None


def _dispatch(pending: list, use_docling: bool) -> tuple[dict, int]:
    """Sort the work into lanes, and count how much of it will reach the card.

    Two numbers out of one inspection per document, because they answer two
    different questions and only one of them is the lane. A lane says where a
    document waits. It does not say which engine resolves it, and the hybrid
    engine reaches the model from the processor lane.

    Reading the first as an answer to the second is the defect this returns a
    second number for: with the card's lane correctly empty, the card's share
    was computed as zero and fell to one process, while 28% of the documents
    used the card anyway.

    Without the structured engine nothing reaches the card at all, so there is
    no inspection to do and no lane to choose.
    """
    lanes: dict[str, list] = {LANE_GPU: [], LANE_CPU: []}
    card_bound = 0
    for job in pending:
        if use_docling:
            lane, reaches_card = inspect_job(job)
        else:
            lane, reaches_card = LANE_CPU, False
        lanes[lane].append(job)
        if reaches_card:
            card_bound += 1
    return lanes, card_bound


# What a process holds on the card by having loaded the models, without
# calculating anything. It is paid on loading, so a process pays it whether or
# not the gate ever lets it compute -- which is the whole of the defect this
# number exists to bound.
#
# Not the 484 MiB a single process shows in isolation. Solved instead from two
# measured runs on the same 6 GB card, which is the only way to separate the
# two costs:
#
#     2 computing + 5 resident = 5,884 MiB   (stopped)
#     3 computing + 3 resident = 5,800 MiB   (finished, at the limit)
#
# giving 1,261 MiB for a process that is computing and 672 for one that has
# merely loaded. The isolated figure undercounts because it does not include
# what the context and the allocator keep per process once several coexist.
VRAM_RESIDENT_MIB = 672

# The former spelling of the per-worker peak, kept readable alongside the new
# one; the resident figure has its own so a card whose models measure
# differently can say so without touching the peak.
RESIDENT_ENV_VARS = ("MDCX_VRAM_RESIDENT",)

# How much of the free video memory the sizing may commit. The remainder is the
# difference between working and thrashing rather than slack: measured on a
# 6 GB card, a run that filled 96% of it stopped advancing altogether, and one
# that filled 88% finished and was slightly faster than one at 95%.
VRAM_USABLE_SHARE = 0.85


def _vram_resident_mib() -> int:
    for name in RESIDENT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            return VRAM_RESIDENT_MIB
    return VRAM_RESIDENT_MIB


def _resident_cap(free: int, gpu_workers: int) -> int:
    """How many processes may hold models on the card at once.

    The processes that hold the gate need room to compute; whatever is left
    over is what the others may occupy merely by having loaded. Both are
    counted because both are on the card at the same time.
    """
    usable = int(free * VRAM_USABLE_SHARE)
    for_turns = gpu_workers * _vram_per_worker_mib()
    spare = usable - for_turns
    return gpu_workers + max(0, spare // _vram_resident_mib())


def _lane_sizes(total_cap: int, gpu_fraction: float, has_gpu: bool) -> tuple[int, int]:
    """How many workers each lane gets, from the machine and from the work.

    Four ceilings, and the smallest wins. All four are needed, and leaving any
    of them out breaks the sizing somewhere:

    The card. Free video memory divided by what a worker holds once its batch is
    full. Without this, raising the count by hand does the opposite of what it
    looks like: twelve workers on a 6 GB card ask for 15.7 GB and the run
    measured three times slower than three workers.

    The work. A lane sized for documents that never arrive is a lane of idle
    processes. What is counted is the work expected to reach the card, which is
    not the same as the size of the card's lane: a document read from its own
    text layer waits in the processor lane and still hands the model the pages
    that announce a table. Counting the lane instead made this share collapse
    to one process the moment the lane was correctly emptied.

    The rest of the machine. Without a third ceiling the formula breaks exactly
    where there is most to gain: a card of 11 GB would allow seven workers on the
    card and leave one for everything else, which is backwards, the bulk of this
    work being processor work.

    What is resident on the card, which bounds the processor lane rather than
    the card's own. The gate says how many processes may compute; it says
    nothing about how many may load, and a process pays for the models the
    moment it builds the converter -- before it ever asks for a turn. Since the
    hybrid engine is invoked from the processor lane, every process in that lane
    ends up holding a CUDA context and a set of models. Measured: seven resident
    on a 6 GB card filled it and the run stopped dead, 76 chapters in 109 s and
    then none in 58 minutes; five finished the same work and were faster than
    six. So the processor lane is capped by what the card can hold, not only by
    what is left of the processor count.

    That last one is the reverse of what intuition says. Because the processor
    lane is the remainder, work that barely needs the card gets the most
    processes loading models into it, and fewest turns to use them: the easiest
    material is the likeliest to stall.
    """
    import math

    if total_cap <= 1:
        return 1, 1

    free = _free_vram_mib() if has_gpu else None

    by_gpu = total_cap
    if free is not None:
        by_gpu = free // _vram_per_worker_mib()

    by_demand = math.ceil(total_cap * max(0.0, min(1.0, gpu_fraction)))
    gpu = max(1, min(by_gpu, by_demand, total_cap - 1))

    cpu = total_cap - gpu
    if free is not None:
        cpu = min(cpu, _resident_cap(free, gpu))
    return gpu, max(1, cpu)


def _worker_budget(threads: int, gate, batch: int | None = None) -> None:
    """Start a worker that keeps to its share of the machine.

    Two shares, and they are not the same thing.

    The processors: counting processes is not enough to know what a run will
    take, because Docling asks for four threads of its own. Eight workers taking
    that at face value ask for thirty-two threads on a machine of twelve, and
    the reserve --max-cores was careful to leave is spent again inside the pool.
    The cap is therefore divided among the workers and handed to each as a
    budget. The variables are read by the numerical libraries when they first
    load, and a worker loads them after this runs, which is why they are set
    here rather than once the process is already busy.

    The card: bounded by a gate every worker shares, rather than by which pool
    the worker belongs to. Both lanes may use the card and both are counted
    against the same limit, so the lane is left deciding what it is good at --
    which documents are worth dispatching where -- and not, as it used to,
    which engines a document is allowed to reach at all.

    The batch comes with it, because the two are the same decision seen from
    either end: how much of the card a worker may hold depends on how many
    workers may hold it. A worker deciding that for itself reads the free
    memory as though it were the only one, and three workers each taking a
    third too much is how a card ends at 96% with the run fighting itself.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    if batch:
        os.environ["MDCX_TATR_BATCH"] = str(batch)
    engines.set_gpu_gate(gate)


def _select_start_method() -> str:
    """Choose how worker processes are started, and say so.

    On Linux, multiprocessing forks by default, and a CUDA context does not
    survive fork(): docling fails inside every child, the engine selector reads
    that failure as "this output is not usable", and the native engine wins. The
    run reports success and full coverage while every table has been flattened
    into a paragraph.

    Windows already spawns, which is why the defect never appeared there.
    """
    import multiprocessing

    current = multiprocessing.get_start_method(allow_none=True)
    if sys.platform == "win32" or current == "spawn":
        return current or "spawn"

    try:
        import torch

        if not torch.cuda.is_available():
            return current or "fork"
    except Exception:  # noqa: BLE001
        return current or "fork"

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already started elsewhere; nothing to do but report what is in use.
        return multiprocessing.get_start_method()
    return "spawn"


def main() -> int:
    console.configure()
    ap = argparse.ArgumentParser(
        description="Convert documents to Markdown with fidelity verification."
    )
    from . import __version__
    ap.add_argument("--version", action="version", version=f"mdcx {__version__}")
    ap.add_argument("--input", default=str(ROOT / "Input"), help="source folder")
    ap.add_argument("--output", default=str(ROOT / "Output"), help="target folder")
    ap.add_argument("--max-cores", type=int, default=MAX_CORES_DEFAULT,
                    help=f"cap on concurrent processes; the default leaves {int(RESERVED_SHARE * 100)}%% of the machine free ({MAX_CORES_DEFAULT} here)")
    ap.add_argument("--gpu-workers", type=int, default=None,
                    help="workers allowed on the card; the default is the smallest of what the video memory fits, what the material asks for, and leaving one elsewhere")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="workers in the other lane (default: the remainder of the cap)")
    ap.add_argument("--serial", action="store_true",
                    help="one document at a time, without parallelism (for clean measurement)")
    ap.add_argument("--only", default=None, help="glob pattern over the file name")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N files")
    ap.add_argument("--force", action="store_true", help="ignore the cache and reconvert everything")
    ap.add_argument("--no-docling", action="store_true", help="use native engines only")
    ap.add_argument("--no-gpu", action="store_true",
                    help="do not use the GPU even if available (all work on CPU)")
    ap.add_argument("--no-lossless", action="store_true", help="do not write the backup JSON")
    ap.add_argument("--no-compact", action="store_true",
                    help="write Markdown with converter scaffolding, without compacting")
    ap.add_argument("--no-split", action="store_true",
                    help="do not split long documents into chapters")
    ap.add_argument("--split-threshold", type=int, default=chapters.SPLIT_THRESHOLD_PAGES,
                    help=f"page count above which to split (default {chapters.SPLIT_THRESHOLD_PAGES})")
    ap.add_argument("--recycle-after", type=int, default=RECYCLE_AFTER_DOCUMENTS,
                    metavar="N",
                    help=f"replace a worker after this many documents, so a "
                         f"thread left running by an abandoned one does not "
                         f"hold a core for the rest of the run "
                         f"(default {RECYCLE_AFTER_DOCUMENTS}; 0 never)")
    ap.add_argument("--timeout", type=float, default=DOCUMENT_TIMEOUT_SECONDS,
                    metavar="SECONDS",
                    help=f"give up on a document after this long and go on to "
                         f"the next (default {DOCUMENT_TIMEOUT_SECONDS:.0f}; "
                         f"0 waits indefinitely)")
    args = ap.parse_args()

    _configure_tls()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        print(f"ERROR: input folder does not exist: {input_root}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped = plan_jobs(input_root)
    if args.only:
        jobs = [j for j in jobs if fnmatch.fnmatch(j.rel_source.name.lower(), args.only.lower())]
    planned = len(jobs)
    if args.limit:
        jobs = jobs[: args.limit]
    # A run cut short by --limit finishes exactly as asked and still leaves work
    # behind, so the progress file survives -- and a reader cannot tell it from
    # a run that died. Both facts are recorded rather than the file being
    # removed, which would lose the ability to resume a limited run that failed
    # halfway.
    limited = bool(args.limit) and len(jobs) < planned

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    use_docling = not args.no_docling and engines.docling_available()
    has_gpu = (not args.no_gpu) and engines.gpu_available()
    device = engines.device_name() if has_gpu else "CPU"

    # The start method is chosen before any pool is created: on Linux a CUDA
    # context does not survive fork(), which silently disabled docling.
    start_method = _select_start_method()

    # The lane sizes are settled further down, once the documents have been
    # classified: how many of them need the card is one of the three things that
    # decides, and it cannot be known before they are looked at.
    print(f"Source : {input_root}")
    print(f"Target : {output_root}")
    print(f"Files to convert: {len(jobs)} | skipped by archive: {len(skipped)}")
    print(f"Structured engine (Docling): {'si' if use_docling else 'no'} | "
          f"inference: {device}")

    cache = {} if args.force else _load_cache(output_root)

    def _already_converted(job) -> dict | None:
        """Reusable previous work for this target, or None if it must be redone."""
        prev = cache.get(to_pseudopath(job.rel_target))
        if not prev or not prev.get("digest") or not prev.get("ok"):
            return None
        if not (output_root / job.rel_target).exists():
            return None
        try:
            if prev["digest"] != file_digest(job.source):
                return None
        except Exception:
            return None
        return prev

    def _adjust_compaction(target: Path, wants_compaction: bool) -> str:
        """Bring an existing .md to the requested compaction state."""
        try:
            current = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "rehacer"

        compacted, applied = compact_module.compact(current)
        already_compacted = applied and compacted == current

        if wants_compaction:
            if already_compacted:
                return "igual"
            if not applied:
                return "igual"
            try:
                target.write_text(compacted, encoding="utf-8")
            except OSError:
                return "rehacer"
            return "ajustado"

        return "rehacer" if already_compacted else "igual"

    documents_split: dict[str, dict] = {}
    units: list = []
    if not args.no_split:
        for job in jobs:
            caps = chapters.plan_chapters(job.source, args.split_threshold) if job.kind == "pdf" else []
            if len(caps) < 2:
                units.append(job)
                continue
            documents_split[to_pseudopath(job.rel_target)] = {
                "document": job.rel_source.name,
                "pseudopath_index": to_pseudopath(job.rel_target),
                "chapters_folder": to_pseudopath(job.rel_target.with_suffix("")),
                "chapters": len(caps),
                # What the original had, beside what the chapters cover. The
                # two are not the same number and the difference is the whole
                # point: a document split into chapters that do not span it is
                # a truncated conversion, and with only the covered count there
                # is no way to tell -- someone deleted seven originals believing
                # the conversion was whole. The count is already known here.
                "pages_total": _pdf.count_pages(job.source),
                "from_pdf_outline": caps[0].from_toc,
                "rel_target": job.rel_target,
                "rel_source": job.rel_source,
                "children": [],
            }
            units.extend(make_chapter_jobs(job, caps))
        if documents_split:
            print(f"Documents split into chapters: {len(documents_split)} "
                  f"-> {sum(d['chapters'] for d in documents_split.values())} chapters")
    else:
        units = list(jobs)

    pending, reused = [], []
    adjusted = 0
    redo_for_compaction = 0
    for job in units:
        prev = _already_converted(job)
        if prev is None:
            pending.append(job)
            continue
        status = _adjust_compaction(output_root / job.rel_target,
                                     not args.no_compact)
        if status == "rehacer":
            redo_for_compaction += 1
            pending.append(job)
            continue
        if status == "ajustado":
            adjusted += 1
            prev = dict(prev)
            prev["compacted"] = not args.no_compact
        reused.append(prev)

    if reused:
        print(f"Already converted and unchanged: {len(reused)} (reused)")
    if adjusted:
        print(f"Compacted without reconversion: {adjusted} "
              "(content was already correct; only scaffolding was removed)")
    if redo_for_compaction:
        print(f"Reconverted due to a change in compaction: {redo_for_compaction} "
              "(compaction cannot be undone without returning to the original)")
    if reused and not pending:
        print("Nothing pending: the output is up to date.")

    lanes, card_bound = _dispatch(pending, use_docling)

    for lane in lanes:
        lanes[lane].sort(key=estimated_cost)

    heavy = [j for j in lanes[LANE_GPU] if estimated_cost(j) >= 100]
    if heavy:
        print(f"Long documents (dispatched last): "
              + ", ".join(f"{j.rel_source.name[:40]} ({estimated_cost(j):.0f} pag)"
                          for j in heavy[-4:]))
    # The start method is chosen before any pool is created: on Linux a CUDA
    # context does not survive fork(), which silently disabled docling.
    start_method = _select_start_method()

    if args.serial:
        gpu_workers, cpu_workers = 1, 1
        print("Serial mode: one document at a time (timings are comparable across runs)")
        threads_per_worker = max(1, args.max_cores)
    else:
        # Of the work, not of the lane: what is expected to reach the card
        # over everything there is to convert.
        gpu_fraction = card_bound / max(1, len(pending))
        gpu_workers, cpu_workers = _lane_sizes(args.max_cores, gpu_fraction, has_gpu)
        # The formula is the default, not a ruling: a machine that knows better
        # says so, and is then held only to the total.
        if args.gpu_workers is not None:
            gpu_workers = max(1, min(args.gpu_workers, args.max_cores - 1))
            cpu_workers = max(1, args.max_cores - gpu_workers)
            # Raising this by hand is the first thing anyone tries, and it does
            # the opposite of what it looks like when the card cannot hold them:
            # measured, twelve workers on a 6 GB card ran three times slower
            # than three, because the resource running out was not the one being
            # raised. Saying so costs a line and saves the measurement.
            free = _free_vram_mib() if has_gpu else None
            fit = (free // _vram_per_worker_mib()) if free else None
            if fit is not None and gpu_workers > max(1, fit):
                print(f"Warning: {gpu_workers} processes were asked of the card "
                      f"and {max(1, fit)} fit ({free} MiB free, "
                      f"{_vram_per_worker_mib()} per process). Past that they "
                      f"compete for memory and the run gets slower, not "
                      f"faster.")
        if args.cpu_workers is not None:
            cpu_workers = max(1, min(args.cpu_workers, args.max_cores - gpu_workers))

        # The batch is settled here, where both halves are known: how many
        # workers may hold the card, and what is left over once they are all
        # seated on it. A worker asked to decide alone reads the free memory as
        # though nobody else would.
        from .convert import tatr as _tatr

        batch = _tatr.batch_for(_free_vram_mib() if has_gpu else None, gpu_workers)

        # The cap is a budget for the whole run, so it is divided among the
        # workers rather than granted to each: otherwise the share it was
        # careful to leave free is spent again inside every process.
        threads_per_worker = max(1, args.max_cores // max(1, gpu_workers + cpu_workers))
        # How many processes may end up holding models on the card. A lane with
        # no documents starts no processes, and both lanes reach the model, so
        # this is not the same number as the turns -- reading the turns as what
        # the card will see is what hid a run filling the card and stopping.
        resident = ((gpu_workers if lanes[LANE_GPU] else 0)
                    + (cpu_workers if lanes[LANE_CPU] else 0))
        print(f"GPU lane: {len(lanes[LANE_GPU])} documents in {gpu_workers} processes | "
              f"CPU lane: {len(lanes[LANE_CPU])} documents in {cpu_workers} processes | "
              f"cap {args.max_cores} of {os.cpu_count() or '?'} cores, "
              f"{threads_per_worker} thread(s) each | "
              f"card: {card_bound} document(s) expected to reach it, "
              f"{resident} process(es) with models resident, up to "
              f"{gpu_workers} computing at a time, batch {batch}")
        if start_method == "spawn" and sys.platform != "win32":
            print("Workers start with spawn: a CUDA context does not survive fork, "
                  "which would disable docling in every child.")
    print()

    started = time.time()
    records: list[dict] = list(reused)
    total = len(pending)
    done = 0

    progress_path = output_root / PROGRESS_FILENAME
    if args.force and progress_path.exists():
        progress_path.unlink(missing_ok=True)
    progress_fh = progress_path.open("a", encoding="utf-8")

    def _settle(rec: dict) -> None:
        try:
            progress_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            progress_fh.flush()
            os.fsync(progress_fh.fileno())
        except Exception:
            pass

    if total:
        if args.serial:
            for lane in (LANE_GPU, LANE_CPU):
                for job in lanes[lane]:
                    # One process, so there is nothing to contend for and no
                    # reason to withhold the engine from either lane.
                    allow_docling = use_docling
                    try:
                        rec = convert_one_safe(
                            (job, output_root, allow_docling, not args.no_lossless,
                             not args.no_compact, args.timeout)
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec = _error_record(job, exc)
                    records.append(rec)
                    _settle(rec)
                    done += 1
                    _print_progress(done, total, lane, rec)
        else:
            import multiprocessing

            # How many processes may hold the card at once. The GPU lane used to
            # be this number by being the only lane with the engines; now it is
            # said once and applies to every worker, whichever lane it is in.
            gate = multiprocessing.Semaphore(gpu_workers)

            recycle = args.recycle_after or None
            with ProcessPoolExecutor(
                    max_workers=gpu_workers, initializer=_worker_budget,
                    max_tasks_per_child=recycle,
                    initargs=(threads_per_worker, gate, batch)) as gpu_pool, \
                 ProcessPoolExecutor(
                    max_workers=cpu_workers, initializer=_worker_budget,
                    max_tasks_per_child=recycle,
                    initargs=(threads_per_worker, gate, batch)) as cpu_pool:
                futures = {}
                for job in lanes[LANE_GPU]:
                    fut = gpu_pool.submit(
                        convert_one_safe, (job, output_root, use_docling, not args.no_lossless, not args.no_compact,
                         args.timeout)
                    )
                    futures[fut] = (job, LANE_GPU)
                for job in lanes[LANE_CPU]:
                    fut = cpu_pool.submit(
                        convert_one_safe,
                        (job, output_root, use_docling, not args.no_lossless, not args.no_compact,
                         args.timeout)
                    )
                    futures[fut] = (job, LANE_CPU)

                for fut in as_completed(futures):
                    job, lane = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = _error_record(job, exc)
                    records.append(rec)
                    _settle(rec)
                    done += 1
                    _print_progress(done, total, lane, rec)

    if documents_split:
        by_document: dict[str, list[dict]] = {}
        for rec in records:
            cap = rec.get("chapter")
            if cap and cap.get("document_pseudopath"):
                rec["is_chapter"] = True
                by_document.setdefault(cap["document_pseudopath"], []).append(rec)
        for pseudo, info in documents_split.items():
            caps = by_document.get(pseudo, [])
            if not caps:
                continue
            try:
                records.append(index.write_document_index(info, caps, output_root))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not write the index for {info['document']}: {exc}")

    try:
        progress_fh.close()
    except Exception:
        pass

    elapsed = time.time() - started
    manifest = index.build_manifest(records, input_root, skipped, elapsed)
    manifest["run"] = {
        "mode": "serial" if args.serial else "parallel",
        "device": device,
        "gpu_processes": gpu_workers,
        "cpu_processes": cpu_workers,
        "core_cap": args.max_cores,
        "documents_gpu_lane": len(lanes[LANE_GPU]),
        "documents_cpu_lane": len(lanes[LANE_CPU]),
    }
    index_path = index.write_index(records, output_root, manifest)
    results_path = index.write_results_txt(records, output_root, manifest, elapsed, input_root)

    res = manifest["summary"]
    print("\n" + "=" * 72)
    print(f"Documents       : {res['documents']}")
    print(f"Conforming      : {res['converted_ok']}")
    print(f"With findings   : {res['with_findings']}")
    if res.get('unverifiable'):
        print(f"Unverifiable    : {res['unverifiable']} "
              "(no text in the original to measure against)")
    cg = res["global_token_coverage"]
    print(f"Global coverage : {cg * 100:.3f}%" if cg is not None else "Global coverage : n/a")
    print(f"Tokens not recovered: {res['tokens_not_recovered']} of {res['reference_tokens']}")
    print(f"Elapsed         : {elapsed / 60:.1f} min ({'serial' if args.serial else 'parallel'})")
    print(f"Index           : {index_path}")
    print(f"Results         : {results_path}")

    if limited:
        print(f"Partial run     : {len(jobs)} of {planned} files, as asked by "
              f"--limit; the progress file is kept and says so")
        try:
            with progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run": "complete",
                                     "limited_to": len(jobs),
                                     "of": planned}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass
    else:
        try:
            progress_path.unlink(missing_ok=True)
        except Exception:
            pass
    print("=" * 72)
    return 0 if res["with_findings"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
