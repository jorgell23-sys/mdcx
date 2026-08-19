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

"""Convierte una carpeta completa de documentos a Markdown, espejando su estructura.

Uso tipico:
    python tool/run.py                      # Input/ -> Output/
    python tool/run.py --only "*.xlsx"      # solo un subconjunto
    python tool/run.py --force              # reconvertir todo, ignorando la cache
    python tool/run.py --serie              # un documento por vez (para medir sin ruido)

Ejecucion en dos carriles simultaneos:

  - Carril GPU: documentos donde el motor con modelos de layout y tablas aporta estructura
    (informes, DOCX, XLSX). Pocos procesos, porque comparten la memoria de la placa.
  - Carril CPU: documentos que se resuelven con extractores deterministas (planos, texto).
    Muchos procesos, porque son livianos y no tocan la GPU.

Los dos avanzan a la vez en vez de competir por los mismos nucleos: mientras la placa
procesa un informe con tablas, la CPU despacha los planos. El total de procesos esta
acotado por --max-cores.
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



from .convertir import chapters, engines, index  # noqa: E402
from .convertir import compactar as compactar_md  # noqa: E402
from .convertir.convert import convert_one_safe  # noqa: E402
from .convertir.paths import (  # noqa: E402
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

# Tope de procesos concurrentes. Por encima de este numero el rendimiento deja de mejorar
# y solo aumenta la contencion, dejando la maquina inutilizable para otras tareas.
MAX_CORES_DEFAULT = 8


def _configure_tls() -> None:
    """Usa el bundle CA local si existe.

    En equipos con antivirus o proxy que inspecciona TLS (Avast, Zscaler y similares),
    el certificado que se presenta lo firma una CA local que no esta en certifi, y la
    descarga de modelos falla. El bundle combinado cubre ambos casos, con y sin
    inspeccion activa, sin desactivar la verificacion.
    """
    if CERT_BUNDLE.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(var, str(CERT_BUNDLE))

    # Con los modelos ya en disco, se corta todo acceso a la red: la conversion no debe
    # depender de un servicio externo ni consultarlo para comprobar versiones. Asi el
    # proceso es reproducible, funciona en un equipo aislado y no puede incurrir en costo.
    if engines.local_artifacts_path() is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Registro que se escribe documento a documento, apenas cada uno termina. El manifiesto
# final solo existe si la corrida llego al final; este archivo es lo que permite retomar
# un lote interrumpido sin rehacer lo ya convertido.
PROGRESS_FILENAME = "_progreso.jsonl"


def _load_cache(output_root: Path) -> dict:
    """Trabajo ya realizado, indexado por el pseudopath del Markdown de destino.

    Se indexa por destino y no por origen porque un documento extenso se divide en varios
    Markdown que comparten el mismo archivo de origen: indexar por origen haria que los
    capitulos se pisaran entre si y solo se reconociera el ultimo.

    Se leen dos fuentes: el manifiesto de la ultima corrida completa y el registro
    incremental de una corrida que haya quedado a medias. El incremental tiene prioridad
    por ser mas reciente.
    """
    cache: dict = {}

    manifest = output_root / "_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for d in data.get("documentos", []):
                if d.get("markdown_pseudopath"):
                    cache[d["markdown_pseudopath"]] = d
        except Exception:
            pass

    progreso = output_root / PROGRESS_FILENAME
    if progreso.exists():
        try:
            with progreso.open("r", encoding="utf-8") as fh:
                for linea in fh:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        d = json.loads(linea)
                    except Exception:
                        # Una linea truncada por un corte abrupto se descarta sola:
                        # ese documento simplemente se vuelve a convertir.
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
    print(
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Conversion de documentos a Markdown con verificacion de fidelidad."
    )
    ap.add_argument("--input", default=str(ROOT / "Input"), help="carpeta de origen")
    ap.add_argument("--output", default=str(ROOT / "Output"), help="carpeta destino")
    ap.add_argument("--max-cores", type=int, default=MAX_CORES_DEFAULT,
                    help=f"tope de procesos concurrentes (por defecto {MAX_CORES_DEFAULT})")
    ap.add_argument("--gpu-workers", type=int, default=None,
                    help="procesos del carril GPU (por defecto 2 con placa, 3 sin ella)")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="procesos del carril CPU (por defecto, el resto del tope)")
    ap.add_argument("--serie", action="store_true",
                    help="un documento por vez, sin paralelismo (para medir sin ruido)")
    ap.add_argument("--only", default=None, help="patron glob sobre el nombre del archivo")
    ap.add_argument("--limit", type=int, default=0, help="convertir solo los primeros N")
    ap.add_argument("--force", action="store_true", help="ignorar cache y reconvertir todo")
    ap.add_argument("--no-docling", action="store_true", help="usar solo motores nativos")
    ap.add_argument("--sin-gpu", action="store_true",
                    help="no usar la GPU aunque este disponible (todo el trabajo en CPU)")
    ap.add_argument("--no-lossless", action="store_true", help="no guardar el JSON de respaldo")
    ap.add_argument("--sin-compactar", action="store_true",
                    help="escribir el Markdown con el andamiaje del conversor, sin compactar")
    ap.add_argument("--no-split", action="store_true",
                    help="no dividir documentos extensos en capitulos")
    ap.add_argument("--split-threshold", type=int, default=chapters.SPLIT_THRESHOLD_PAGES,
                    help=f"paginas a partir de las cuales dividir (por defecto {chapters.SPLIT_THRESHOLD_PAGES})")
    args = ap.parse_args()

    _configure_tls()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        print(f"ERROR: no existe la carpeta de entrada: {input_root}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped = plan_jobs(input_root)
    if args.only:
        jobs = [j for j in jobs if fnmatch.fnmatch(j.rel_source.name.lower(), args.only.lower())]
    if args.limit:
        jobs = jobs[: args.limit]

    if args.sin_gpu:
        # "-1" es el valor que oculta toda placa; la cadena vacia no siempre se respeta.
        # Se fija antes de crear los procesos worker para que la decision valga tambien
        # dentro de ellos, que es donde se cargan los modelos.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    use_docling = not args.no_docling and engines.docling_available()
    tiene_gpu = (not args.sin_gpu) and engines.gpu_available()
    dispositivo = engines.device_name() if tiene_gpu else "CPU"

    # Reparto de procesos entre carriles, respetando el tope total.
    if args.serie:
        gpu_workers, cpu_workers = 1, 1
    else:
        gpu_workers = args.gpu_workers if args.gpu_workers is not None else (2 if tiene_gpu else 3)
        gpu_workers = max(1, min(gpu_workers, args.max_cores - 1))
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else args.max_cores - gpu_workers
        cpu_workers = max(1, min(cpu_workers, args.max_cores - gpu_workers))

    print(f"Origen : {input_root}")
    print(f"Destino: {output_root}")
    print(f"Archivos a convertir: {len(jobs)} | omitidos por formato: {len(skipped)}")
    print(f"Motor estructurado (Docling): {'si' if use_docling else 'no'} | "
          f"inferencia: {dispositivo}")

    cache = {} if args.force else _load_cache(output_root)

    def _ya_convertido(job) -> dict | None:
        """Trabajo previo aprovechable para este destino, o None si hay que rehacerlo."""
        prev = cache.get(to_pseudopath(job.rel_target))
        if not prev or not prev.get("digest") or not prev.get("ok"):
            return None
        if not (output_root / job.rel_target).exists():
            return None
        # Se compara el contenido del original, no su fecha: copiar un archivo cambia la
        # fecha sin cambiar el contenido, y eso obligaria a reconvertir sin motivo.
        try:
            if prev["digest"] != file_digest(job.source):
                return None
        except Exception:
            return None
        return prev

    def _ajustar_compactado(destino: Path, quiere_compactado: bool) -> str:
        """Pone el .md ya escrito en el estado de compactado que se pidio.

        Devuelve "igual" si ya estaba como corresponde, "ajustado" si se pudo arreglar aqui
        mismo, y "rehacer" si hace falta volver a convertir desde el original.

        Compactar es barato -unos milisegundos de texto- y no necesita el documento de
        origen, asi que cuando falta se hace en el acto en vez de reconvertir un PDF que
        puede tardar minutos. La direccion contraria no tiene atajo: el andamiaje retirado
        no se puede reconstruir, y recuperarlo exige volver al original.
        """
        try:
            actual = destino.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "rehacer"

        compacto, aplicada = compactar_md.compactar(actual)
        ya_compactado = aplicada and compacto == actual

        if quiere_compactado:
            if ya_compactado:
                return "igual"
            if not aplicada:
                # La comprobacion de contenido rechazo la compactacion de este archivo. Se
                # deja como esta: perder una palabra nunca compensa el ahorro.
                return "igual"
            try:
                destino.write_text(compacto, encoding="utf-8")
            except OSError:
                return "rehacer"
            return "ajustado"

        return "rehacer" if ya_compactado else "igual"

    # Division de documentos extensos. Un PDF de 300 paginas ocuparia un proceso durante
    # media hora mientras el resto de la cola espera; partido por capitulos, los tramos se
    # reparten entre todos los procesos y ademas el resultado queda legible.
    #
    # La division se resuelve antes de decidir que falta convertir, porque la unidad real
    # de trabajo es el capitulo: si se interrumpe un documento de 300 paginas a la mitad,
    # se retoman los capitulos que faltan y no el documento entero.
    documentos_divididos: dict[str, dict] = {}
    unidades: list = []
    if not args.no_split:
        for job in jobs:
            caps = chapters.plan_chapters(job.source, args.split_threshold) if job.kind == "pdf" else []
            if len(caps) < 2:
                unidades.append(job)
                continue
            documentos_divididos[to_pseudopath(job.rel_target)] = {
                "documento": job.rel_source.name,
                "pseudopath_indice": to_pseudopath(job.rel_target),
                "carpeta_capitulos": to_pseudopath(job.rel_target.with_suffix("")),
                "capitulos": len(caps),
                "desde_indice_pdf": caps[0].from_toc,
                "rel_target": job.rel_target,
                "rel_source": job.rel_source,
                "hijos": [],
            }
            unidades.extend(make_chapter_jobs(job, caps))
        if documentos_divididos:
            print(f"Documentos divididos por capitulos: {len(documentos_divididos)} "
                  f"-> {sum(d['capitulos'] for d in documentos_divididos.values())} capitulos")
    else:
        unidades = list(jobs)

    # Reparto entre lo que ya esta hecho y lo que falta. Un capitulo convertido en una
    # corrida anterior o interrumpida se reutiliza tal cual.
    pending, reused = [], []
    ajustados = 0
    rehacer_por_compactado = 0
    for job in unidades:
        prev = _ya_convertido(job)
        if prev is None:
            pending.append(job)
            continue
        # La cache decide por el contenido del original, que es lo correcto para saber si
        # hay que volver a convertir. Pero no dice nada sobre el estado del .md ya escrito,
        # y ese estado cambia cuando se cambia la opcion de compactado. Sin esta
        # comprobacion, pedir compactado sobre una carpeta convertida sin el no tenia
        # ningun efecto: la cache reutilizaba y los archivos se quedaban como estaban.
        estado = _ajustar_compactado(output_root / job.rel_target,
                                     not args.sin_compactar)
        if estado == "rehacer":
            rehacer_por_compactado += 1
            pending.append(job)
            continue
        if estado == "ajustado":
            ajustados += 1
            prev = dict(prev)
            prev["compactado"] = not args.sin_compactar
        reused.append(prev)

    if reused:
        print(f"Ya convertidos y sin cambios: {len(reused)} (se reutilizan)")
    if ajustados:
        print(f"Compactados sin reconvertir: {ajustados} "
              "(el contenido ya estaba bien, solo faltaba retirar el andamiaje)")
    if rehacer_por_compactado:
        print(f"Se reconvierten por cambio de compactado: {rehacer_por_compactado} "
              "(compactar no se puede deshacer sin volver al original)")
    if reused and not pending:
        print("No hay nada pendiente: la salida ya esta al dia.")

    # Asignacion de carril. Con --no-docling todo se resuelve con extractores nativos.
    lanes: dict[str, list] = {LANE_GPU: [], LANE_CPU: []}
    for job in pending:
        lane = LANE_CPU if not use_docling else classify_lane(job)
        lanes[lane].append(job)

    # Despachar de menor a mayor coste. Sin este orden, un par de documentos de 300 paginas
    # al frente de la cola ocupa todos los procesos durante media hora mientras decenas de
    # documentos de una pagina esperan detras sin motivo.
    for lane in lanes:
        lanes[lane].sort(key=estimated_cost)

    pesados = [j for j in lanes[LANE_GPU] if estimated_cost(j) >= 100]
    if pesados:
        print(f"Documentos extensos (se despachan al final): "
              + ", ".join(f"{j.rel_source.name[:40]} ({estimated_cost(j):.0f} pag)"
                          for j in pesados[-4:]))

    if args.serie:
        print("Modo serie: un documento por vez (los tiempos son comparables entre corridas)")
    else:
        print(f"Carril GPU: {len(lanes[LANE_GPU])} documentos en {gpu_workers} procesos | "
              f"Carril CPU: {len(lanes[LANE_CPU])} documentos en {cpu_workers} procesos | "
              f"tope {args.max_cores} cores")
    print()

    started = time.time()
    records: list[dict] = list(reused)
    total = len(pending)
    done = 0

    # Registro incremental: cada documento terminado se asienta en disco de inmediato.
    # Si la corrida se corta (cierre, corte de luz, cancelacion), lo hecho queda anotado
    # y la proxima ejecucion retoma desde ahi en vez de empezar de cero.
    progreso_path = output_root / PROGRESS_FILENAME
    if args.force and progreso_path.exists():
        progreso_path.unlink(missing_ok=True)
    progreso_fh = progreso_path.open("a", encoding="utf-8")

    def _asentar(rec: dict) -> None:
        try:
            progreso_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Se fuerza el volcado a disco: un buffer sin vaciar se pierde con el proceso,
            # que es justo el escenario para el que existe este registro.
            progreso_fh.flush()
            os.fsync(progreso_fh.fileno())
        except Exception:
            pass

    if total:
        if args.serie:
            # Sin pools ni concurrencia: el tiempo por documento no compite con nada.
            for lane in (LANE_GPU, LANE_CPU):
                for job in lanes[lane]:
                    permitir_docling = use_docling and lane == LANE_GPU
                    try:
                        rec = convert_one_safe(
                            (job, output_root, permitir_docling, not args.no_lossless,
                             not args.sin_compactar)
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec = _error_record(job, exc)
                    records.append(rec)
                    _asentar(rec)
                    done += 1
                    _print_progress(done, total, lane, rec)
        else:
            # Los dos pools se lanzan juntos y avanzan en paralelo: el carril CPU no espera
            # a que la GPU termine, y viceversa.
            with ProcessPoolExecutor(max_workers=gpu_workers) as gpu_pool, \
                 ProcessPoolExecutor(max_workers=cpu_workers) as cpu_pool:
                futures = {}
                for job in lanes[LANE_GPU]:
                    fut = gpu_pool.submit(
                        convert_one_safe, (job, output_root, use_docling, not args.no_lossless, not args.sin_compactar)
                    )
                    futures[fut] = (job, LANE_GPU)
                for job in lanes[LANE_CPU]:
                    # El carril CPU no carga modelos: se le indica explicitamente que use
                    # los extractores nativos, para no reservar memoria de la placa.
                    fut = cpu_pool.submit(
                        convert_one_safe, (job, output_root, False, not args.no_lossless, not args.sin_compactar)
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

    # Consolidacion de los documentos divididos: con todos sus capitulos convertidos, se
    # escribe el .md que los agrupa y se agrega un registro del documento completo, para
    # que el indice general siga mostrando una entrada por documento original.
    if documentos_divididos:
        por_documento: dict[str, list[dict]] = {}
        for rec in records:
            cap = rec.get("chapter")
            if cap and cap.get("documento_pseudopath"):
                rec["es_capitulo"] = True
                por_documento.setdefault(cap["documento_pseudopath"], []).append(rec)
        for pseudo, info in documentos_divididos.items():
            caps = por_documento.get(pseudo, [])
            if not caps:
                continue
            try:
                records.append(index.write_document_index(info, caps, output_root))
            except Exception as exc:  # noqa: BLE001
                print(f"AVISO: no se pudo escribir el indice de {info['documento']}: {exc}")

    try:
        progreso_fh.close()
    except Exception:
        pass

    elapsed = time.time() - started
    manifest = index.build_manifest(records, input_root, skipped, elapsed)
    manifest["ejecucion"] = {
        "modo": "serie" if args.serie else "paralelo",
        "dispositivo": dispositivo,
        "procesos_gpu": gpu_workers,
        "procesos_cpu": cpu_workers,
        "tope_cores": args.max_cores,
        "documentos_carril_gpu": len(lanes[LANE_GPU]),
        "documentos_carril_cpu": len(lanes[LANE_CPU]),
    }
    index_path = index.write_index(records, output_root, manifest)
    resultados_path = index.write_results_txt(records, output_root, manifest, elapsed, input_root)

    res = manifest["resumen"]
    print("\n" + "=" * 72)
    print(f"Documentos      : {res['documentos']}")
    print(f"Conformes       : {res['convertidos_ok']}")
    print(f"Con observacion : {res['con_observaciones']}")
    cg = res["cobertura_global_tokens"]
    print(f"Cobertura global: {cg * 100:.3f}%" if cg is not None else "Cobertura global: n/d")
    print(f"Tokens no recuperados: {res['tokens_no_recuperados']} de {res['tokens_referencia']}")
    print(f"Tiempo          : {elapsed / 60:.1f} min ({'serie' if args.serie else 'paralelo'})")
    print(f"Indice          : {index_path}")
    print(f"Resultados      : {resultados_path}")

    # La corrida llego al final y el manifiesto ya contiene todo: el registro incremental
    # cumplio su proposito y se retira para no dejar un archivo de trabajo en la salida.
    try:
        progreso_path.unlink(missing_ok=True)
    except Exception:
        pass
    print("=" * 72)
    return 0 if res["con_observaciones"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
