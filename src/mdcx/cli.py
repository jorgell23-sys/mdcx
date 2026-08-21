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
from .convert import compact as compact_module  # noqa: E402
from .convert.convert import convert_one_safe  # noqa: E402
from .convert.paths import (  # noqa: E402
    LANE_CPU,
    LANE_GPU,
    classify_lane,
    estimated_cost,
    file_digest,
    make_chapter_jobs,
    plan_jobs,
    to_pseudopath,
)

ROOT = Path.cwd()
CERT_BUNDLE = ROOT / "tool" / "certs" / "ca-bundle.pem"

MAX_CORES_DEFAULT = 8

def _configure_tls() -> None:
    """Use the local CA bundle if present."""
    if CERT_BUNDLE.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(var, str(CERT_BUNDLE))

    if engines.local_artifacts_path() is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROGRESS_FILENAME = "_progreso.jsonl"

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

    progreso = output_root / PROGRESS_FILENAME
    if progreso.exists():
        try:
            with progreso.open("r", encoding="utf-8") as fh:
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
    cov_s = f"{cov * 100:6.2f}%" if cov is not None else "  n/d "
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
    ap.add_argument("--input", default=str(ROOT / "Input"), help="source folder")
    ap.add_argument("--output", default=str(ROOT / "Output"), help="target folder")
    ap.add_argument("--max-cores", type=int, default=MAX_CORES_DEFAULT,
                    help=f"tope de procesos concurrentes (por defecto {MAX_CORES_DEFAULT})")
    ap.add_argument("--gpu-workers", type=int, default=None,
                    help="GPU lane workers (default 2 with a card, 3 without)")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="CPU lane workers (default: the remainder of the cap)")
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
                    help=f"page count above which to split (por defecto {chapters.SPLIT_THRESHOLD_PAGES})")
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
    if args.limit:
        jobs = jobs[: args.limit]

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    use_docling = not args.no_docling and engines.docling_available()
    tiene_gpu = (not args.no_gpu) and engines.gpu_available()
    dispositivo = engines.device_name() if tiene_gpu else "CPU"

    # The start method is chosen before any pool is created: on Linux a CUDA
    # context does not survive fork(), which silently disabled docling.
    start_method = _select_start_method()

    if args.serial:
        gpu_workers, cpu_workers = 1, 1
        print("Serial mode: one document at a time (timings are comparable across runs)")
    else:
        gpu_workers = args.gpu_workers if args.gpu_workers is not None else (2 if tiene_gpu else 3)
        gpu_workers = max(1, min(gpu_workers, args.max_cores - 1))
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else args.max_cores - gpu_workers
        cpu_workers = max(1, min(cpu_workers, args.max_cores - gpu_workers))

    print(f"Source : {input_root}")
    print(f"Destino: {output_root}")
    print(f"Files to convert: {len(jobs)} | skipped por archive: {len(skipped)}")
    print(f"Engine estructurado (Docling): {'si' if use_docling else 'no'} | "
          f"inference: {dispositivo}")

    cache = {} if args.force else _load_cache(output_root)

    def _ya_convertido(job) -> dict | None:
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

    def _ajustar_compactado(target: Path, quiere_compactado: bool) -> str:
        """Bring an existing .md to the requested compaction state."""
        try:
            actual = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "rehacer"

        compacto, aplicada = compact_module.compact(actual)
        ya_compactado = aplicada and compacto == actual

        if quiere_compactado:
            if ya_compactado:
                return "igual"
            if not aplicada:
                return "igual"
            try:
                target.write_text(compacto, encoding="utf-8")
            except OSError:
                return "rehacer"
            return "ajustado"

        return "rehacer" if ya_compactado else "igual"

    documents_divididos: dict[str, dict] = {}
    unidades: list = []
    if not args.no_split:
        for job in jobs:
            caps = chapters.plan_chapters(job.source, args.split_threshold) if job.kind == "pdf" else []
            if len(caps) < 2:
                unidades.append(job)
                continue
            documents_divididos[to_pseudopath(job.rel_target)] = {
                "document": job.rel_source.name,
                "pseudopath_indice": to_pseudopath(job.rel_target),
                "carpeta_chapters": to_pseudopath(job.rel_target.with_suffix("")),
                "chapters": len(caps),
                "from_pdf_outline": caps[0].from_toc,
                "rel_target": job.rel_target,
                "rel_source": job.rel_source,
                "hijos": [],
            }
            unidades.extend(make_chapter_jobs(job, caps))
        if documents_divididos:
            print(f"Documentos divididos por chapters: {len(documents_divididos)} "
                  f"-> {sum(d['chapters'] for d in documents_divididos.values())} chapters")
    else:
        unidades = list(jobs)

    pending, reused = [], []
    adjusted = 0
    rehacer_por_compactado = 0
    for job in unidades:
        prev = _ya_convertido(job)
        if prev is None:
            pending.append(job)
            continue
        estado = _ajustar_compactado(output_root / job.rel_target,
                                     not args.no_compact)
        if estado == "rehacer":
            rehacer_por_compactado += 1
            pending.append(job)
            continue
        if estado == "ajustado":
            adjusted += 1
            prev = dict(prev)
            prev["compacted"] = not args.no_compact
        reused.append(prev)

    if reused:
        print(f"Already converted and unchanged: {len(reused)} (reused)")
    if adjusted:
        print(f"Compacted without reconversion: {adjusted} "
              "(content was already correct; only scaffolding was removed)")
    if rehacer_por_compactado:
        print(f"Reconverted due to a change in compaction: {rehacer_por_compactado} "
              "(compaction cannot be undone without returning to the original)")
    if reused and not pending:
        print("Nothing pending: the output is up to date.")

    lanes: dict[str, list] = {LANE_GPU: [], LANE_CPU: []}
    for job in pending:
        lane = LANE_CPU if not use_docling else classify_lane(job)
        lanes[lane].append(job)

    for lane in lanes:
        lanes[lane].sort(key=estimated_cost)

    pesados = [j for j in lanes[LANE_GPU] if estimated_cost(j) >= 100]
    if pesados:
        print(f"Long documents (dispatched last): "
              + ", ".join(f"{j.rel_source.name[:40]} ({estimated_cost(j):.0f} pag)"
                          for j in pesados[-4:]))
    # The start method is chosen before any pool is created: on Linux a CUDA
    # context does not survive fork(), which silently disabled docling.
    start_method = _select_start_method()

    if args.serial:
        gpu_workers, cpu_workers = 1, 1
        print("Serial mode: one document at a time (timings are comparable across runs)")
    else:
        print(f"Carril GPU: {len(lanes[LANE_GPU])} documents en {gpu_workers} procesos | "
              f"Carril CPU: {len(lanes[LANE_CPU])} documents en {cpu_workers} procesos | "
              f"tope {args.max_cores} cores")
        if start_method == "spawn" and sys.platform != "win32":
            print("Workers start with spawn: a CUDA context does not survive fork, "
                  "which would disable docling in every child.")
    print()

    started = time.time()
    records: list[dict] = list(reused)
    total = len(pending)
    done = 0

    progreso_path = output_root / PROGRESS_FILENAME
    if args.force and progreso_path.exists():
        progreso_path.unlink(missing_ok=True)
    progreso_fh = progreso_path.open("a", encoding="utf-8")

    def _asentar(rec: dict) -> None:
        try:
            progreso_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            progreso_fh.flush()
            os.fsync(progreso_fh.fileno())
        except Exception:
            pass

    if total:
        if args.serial:
            for lane in (LANE_GPU, LANE_CPU):
                for job in lanes[lane]:
                    permitir_docling = use_docling and lane == LANE_GPU
                    try:
                        rec = convert_one_safe(
                            (job, output_root, permitir_docling, not args.no_lossless,
                             not args.no_compact)
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec = _error_record(job, exc)
                    records.append(rec)
                    _asentar(rec)
                    done += 1
                    _print_progress(done, total, lane, rec)
        else:
            with ProcessPoolExecutor(max_workers=gpu_workers) as gpu_pool, \
                 ProcessPoolExecutor(max_workers=cpu_workers) as cpu_pool:
                futures = {}
                for job in lanes[LANE_GPU]:
                    fut = gpu_pool.submit(
                        convert_one_safe, (job, output_root, use_docling, not args.no_lossless, not args.no_compact)
                    )
                    futures[fut] = (job, LANE_GPU)
                for job in lanes[LANE_CPU]:
                    fut = cpu_pool.submit(
                        convert_one_safe, (job, output_root, False, not args.no_lossless, not args.no_compact)
                    )
                    futures[fut] = (job, LANE_CPU)

                for fut in as_completed(futures):
                    job, lane = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = _error_record(job, exc)
                    records.append(rec)
                    _asentar(rec)
                    done += 1
                    _print_progress(done, total, lane, rec)

    if documents_divididos:
        por_documento: dict[str, list[dict]] = {}
        for rec in records:
            cap = rec.get("chapter")
            if cap and cap.get("documento_pseudopath"):
                rec["es_capitulo"] = True
                por_documento.setdefault(cap["documento_pseudopath"], []).append(rec)
        for pseudo, info in documents_divididos.items():
            caps = por_documento.get(pseudo, [])
            if not caps:
                continue
            try:
                records.append(index.write_document_index(info, caps, output_root))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not write the index for {info['document']}: {exc}")

    try:
        progreso_fh.close()
    except Exception:
        pass

    elapsed = time.time() - started
    manifest = index.build_manifest(records, input_root, skipped, elapsed)
    manifest["ejecucion"] = {
        "modo": "serial" if args.serial else "parallel",
        "dispositivo": dispositivo,
        "procesos_gpu": gpu_workers,
        "procesos_cpu": cpu_workers,
        "tope_cores": args.max_cores,
        "documents_carril_gpu": len(lanes[LANE_GPU]),
        "documents_carril_cpu": len(lanes[LANE_CPU]),
    }
    index_path = index.write_index(records, output_root, manifest)
    resultados_path = index.write_results_txt(records, output_root, manifest, elapsed, input_root)

    res = manifest["resumen"]
    print("\n" + "=" * 72)
    print(f"Documentos      : {res['documents']}")
    print(f"Conforming      : {res['converted_ok']}")
    print(f"With findings   : {res['with_findings']}")
    if res.get('unverifiable'):
        print(f"Unverifiable    : {res['unverifiable']} "
              "(no text in the original to measure against)")
    cg = res["global_token_coverage"]
    print(f"Global coverage : {cg * 100:.3f}%" if cg is not None else "Global coverage : n/d")
    print(f"Tokens not recovered: {res['tokens_not_recovered']} of {res['reference_tokens']}")
    print(f"Elapsed         : {elapsed / 60:.1f} min ({'serial' if args.serial else 'parallel'})")
    print(f"Index           : {index_path}")
    print(f"Results         : {resultados_path}")

    try:
        progreso_path.unlink(missing_ok=True)
    except Exception:
        pass
    print("=" * 72)
    return 0 if res["with_findings"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
