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

"""Pipeline de conversion de un archivo, con verificacion y recuperacion de faltantes.

Estrategia por archivo:
  1. Leer el texto de referencia con una libreria independiente (extract.py).
  2. Convertir con uno o mas motores (engines.py) y medir la fidelidad de cada salida.
  3. Quedarse con la mejor conversion.
  4. Si aun asi falta contenido, anexar un bloque de recuperacion con exactamente las
     lineas del original que no aparecieron. Asi el .md nunca pierde informacion,
     aunque el motor estructurado se haya equivocado.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import chapters, compactar as compactar_md, engines, extract, verify
from .paths import Job, file_digest, to_pseudopath

# Senales de estructura en el Markdown: titulos, filas separadoras de tabla y vinetas.
_H_RE = re.compile(r"(?m)^#{1,6}\s+\S")
_TBL_RE = re.compile(r"(?m)^\s*\|[-: |]+\|\s*$")
_LI_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")

# Un plano de una sola pagina no tiene jerarquia que Docling pueda mejorar: su texto son
# etiquetas sueltas sobre un dibujo. El motor nativo las conserva igual y es mucho mas
# rapido, asi que se evita pagar el costo del modelo de layout donde no aporta.
PLAN_MAX_PAGES = 2

# Fraccion minima de una linea del original que debe estar ausente para que la linea
# completa se anexe. Evita duplicar texto ya convertido que solo comparte palabras
# comunes con el contenido realmente faltante.
RECOVERY_LINE_RATIO = 0.5


def _yaml_escape(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _front_matter(job: Job, record: dict) -> str:
    v = record["verification"]
    lines = [
        "---",
        f"titulo: {_yaml_escape(job.rel_source.stem)}",
        f"origen: {_yaml_escape(job.rel_source.name)}",
        f"pseudopath_origen: {_yaml_escape(job.source_pseudopath)}",
        f"pseudopath_markdown: {_yaml_escape(job.pseudopath)}",
        f"carpeta: {_yaml_escape(job.rel_source.parent.as_posix() or '.')}",
        f"formato_origen: {job.source.suffix.lower().lstrip('.')}",
        f"bytes_origen: {job.size}",
        f"sha256_origen: {record['digest']}",
        f"motor: {record['engine']}",
        f"ocr: {str(record.get('ocr', False)).lower()}",
        f"convertido_utc: {record['converted_at']}",
        f"fidelidad: {v.get('coverage') if v.get('coverage') is not None else 'no-medible'}",
        f"fidelidad_numerica: {v.get('numeric_coverage') if v.get('numeric_coverage') is not None else 'no-medible'}",
        f"estado_verificacion: {v.get('status')}",
    ]
    if record.get("pages"):
        lines.append(f"paginas: {record['pages']}")
    if record.get("recovered_lines"):
        lines.append(f"lineas_recuperadas: {record['recovered_lines']}")
    lines.append("---")
    return "\n".join(lines)


def _recovery_block(reference: str, markdown: str) -> tuple[str, int]:
    """Construye el anexo con las lineas del original ausentes en el Markdown.

    Se trabaja sobre el deficit de tokens (lo que el original tiene y el Markdown no) y
    se recorren las lineas del original consumiendo ese deficit. Solo entra al anexo la
    linea que aporta tokens realmente faltantes, de modo que no se duplica contenido
    que el motor si convirtio bien.
    """
    ref_tokens = verify.tokenize(reference)
    # Se mide con la misma normalizacion que usa la verificacion. Si el anexo contara el
    # Markdown crudo y la verificacion el normalizado, el anexo creeria cubierto un texto
    # que la verificacion sigue viendo ausente, y el documento quedaria marcado como
    # incompleto sin que nada lo complete nunca.
    md_tokens = verify.tokenize(verify.strip_markdown_noise(markdown))
    deficit = ref_tokens - md_tokens  # Counter resta y descarta los <= 0
    if not deficit:
        return "", 0

    # El deficit se descuenta a mano, termino a termino, en lugar de con la resta de
    # Counter. La resta parece equivalente y no lo es: cada vez que se aplica recorre el
    # contador entero para eliminar las entradas que llegaron a cero. Con un documento de
    # veintitres mil lineas y dos mil terminos distintos eso son cuarenta y cuatro millones
    # de comprobaciones que no cambian nada, y era la causa de que una lista de equipos
    # tardara cuarenta y ocho minutos en convertirse cuando extraerla lleva segundo y medio.
    #
    # Descontando solo las claves de la linea, el trabajo pasa a ser proporcional al texto y
    # no al producto del texto por el vocabulario. El anexo resultante es identico.
    restante = sum(deficit.values())
    recovered: list[str] = []
    for raw_line in reference.splitlines():
        if restante <= 0:
            break
        line = raw_line.strip()
        if not line:
            continue
        tokens = verify.tokenize(line)
        total = sum(tokens.values())
        if not total:
            continue
        hits = sum(min(c, deficit.get(t, 0)) for t, c in tokens.items())
        # Se exige que la linea sea mayoritariamente contenido faltante. Sin este
        # umbral, una linea ya convertida entraria al anexo solo por compartir una
        # palabra comun con la que si falta, duplicando texto sin aportar nada.
        if hits / total < RECOVERY_LINE_RATIO:
            continue
        recovered.append(line)
        for t, c in tokens.items():
            pendiente = deficit.get(t, 0)
            if not pendiente:
                continue
            usados = c if c < pendiente else pendiente
            if usados == pendiente:
                del deficit[t]
            else:
                deficit[t] = pendiente - usados
            restante -= usados

    # Residuo: palabras o cifras sueltas perdidas dentro de lineas que si se convirtieron.
    # No tienen una linea propia que anexar, pero deben quedar registradas para que la
    # informacion del original este completa en el Markdown.
    fragments = sorted(deficit.elements()) if deficit else []

    if not recovered and not fragments:
        return "", 0

    parts = [
        "\n\n---\n\n",
        "## Anexo de recuperacion\n\n",
        "> Contenido presente en el documento original que el motor de conversion "
        "estructurada no incluyo. Se anexa literal, sin reformatear, para garantizar "
        "que el Markdown no pierda informacion respecto de la fuente.\n\n",
    ]
    if recovered:
        parts.append("```text\n" + "\n".join(recovered) + "\n```\n")
    if fragments:
        parts.append(
            "\n**Fragmentos sueltos no ubicados en una linea propia:**\n\n"
            "```text\n" + " ".join(fragments) + "\n```\n"
        )
    return "".join(parts), len(recovered)


def _is_drawing(job: Job, ref_meta: dict) -> bool:
    """Reconoce un plano o diagrama: pocas paginas, imagenes y texto disperso.

    Medido sobre los PFD del corpus, el motor con modelo de layout tarda ~18 s por plano,
    recupera solo el 80% del texto y no produce ni un titulo ni una tabla: el dibujo no
    tiene jerarquia que reconocer, solo etiquetas sueltas sobre una lamina. El extractor
    crudo devuelve el 100% del texto en decimas de segundo. Se evita el motor pesado
    donde la evidencia muestra que empeora el resultado.
    """
    if job.kind != "pdf":
        return False
    pages = ref_meta.get("pages") or 0
    if not 0 < pages <= PLAN_MAX_PAGES:
        return False
    if ref_meta.get("embedded_images", 0) < 1:
        return False
    return True


def _candidates(job: Job, ref_meta: dict, use_docling: bool) -> list[tuple[str, callable]]:
    """Decide que motores probar y en que orden, segun el tipo de documento."""
    native = engines.NATIVE.get(job.kind)
    needs_ocr = bool(ref_meta.get("needs_ocr"))

    if job.kind == "text":
        return [("nativo", lambda p: engines.native_text(p))]

    out: list[tuple[str, callable]] = []
    docling_ok = use_docling and engines.docling_available()

    if docling_ok and needs_ocr:
        # Paginas sin texto embebido: reconocerlas es la unica via de lectura posible.
        out.append(("docling-ocr", lambda p: engines.docling_convert(p, ocr=True)))

    if _is_drawing(job, ref_meta) and not needs_ocr:
        if native:
            out.append(("nativo", native))
        return out

    if docling_ok:
        out.append(("docling", lambda p: engines.docling_convert(p, ocr=False)))
    if native:
        out.append(("nativo", native))
    return out


# Un motor que solo trajo una fraccion del texto no se considera una conversion valida
# aunque el anexo pueda completarla: el resultado seria un titulo suelto seguido de un
# volcado plano. Por debajo de este umbral se prefiere el extractor crudo, que al menos
# conserva el orden de lectura del documento.
MIN_USABLE_COVERAGE = 0.60

# Minimo de palabras que debe traer un documento sin texto extraible para considerar que
# el reconocimiento optico funciono. Por debajo, lo unico que hay en el Markdown son las
# marcas de pagina que agrega el propio conversor.
MIN_TOKENS_SIN_REFERENCIA = 15


def _score(md: str, v: dict) -> tuple:
    """Ordena candidatos: primero los que trajeron el texto, luego los que dan estructura.

    La cobertura sola no sirve para elegir: el extractor crudo usa la misma libreria que
    la lectura de referencia, asi que puntua ~100% por construccion, y ganaria siempre
    entregando texto sin titulos ni tablas. Como el anexo de recuperacion garantiza que
    cualquier candidato viable termine sin perdida, el criterio de desempate real es
    cuanta estructura aporta el Markdown.
    """
    cov = v.get("coverage")
    usable = 1 if (cov is None and md.strip()) or (cov is not None and cov >= MIN_USABLE_COVERAGE) else 0
    structure = len(_H_RE.findall(md)) * 2 + len(_TBL_RE.findall(md)) * 5 + len(_LI_RE.findall(md))
    return (usable, structure, cov or 0.0, len(md))


def convert_one(job: Job, output_root: Path, use_docling: bool = True,
                save_lossless: bool = True, compactar: bool = True) -> dict:
    """Convierte un archivo y escribe su .md espejo. Devuelve el registro para el indice."""
    started = time.time()
    record: dict = {
        "source_pseudopath": job.source_pseudopath,
        "markdown_pseudopath": job.pseudopath,
        "source_name": job.rel_source.name,
        "folder": job.rel_source.parent.as_posix() or ".",
        "format": job.source.suffix.lower().lstrip("."),
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

    # Un capitulo se convierte a partir de un PDF temporal que contiene solo su rango de
    # paginas. Se recorta antes de leer la referencia para que la verificacion de fidelidad
    # compare el capitulo contra su propio tramo del original, no contra el documento entero.
    fuente = job.source
    temporal: Path | None = None
    if job.is_chapter:
        try:
            temporal = Path(tempfile.gettempdir()) / (
                f"pdftomd_{os.getpid()}_{job.chapter_index:03d}_{job.source.stem[:40]}.pdf"
            )
            chapters.extract_range(job.source, job.page_range[0], job.page_range[1], temporal)
            fuente = temporal
        except Exception as exc:  # noqa: BLE001
            record["errors"].append(f"recorte de paginas: {type(exc).__name__}: {exc}")
            temporal = None

    record["chapter"] = {
        "titulo": job.chapter_title,
        "indice": job.chapter_index,
        "paginas": list(job.page_range),
        "documento_pseudopath": to_pseudopath(job.parent_target) if job.parent_target else None,
    } if job.is_chapter else None

    reference, ref_meta = extract.reference_text(fuente, job.kind)
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
            md, meta, lossless = fn(fuente)
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

        if best_score is None or score > best_score:
            best_md, best_v, best_engine, best_score = md, v, name, score
            best_lossless = lossless
            best_ocr = bool(meta.get("ocr"))
            if meta.get("pages"):
                record.setdefault("pages", meta["pages"])

        # Docling conforme y con estructura: los motores siguientes no pueden mejorarlo.
        if name.startswith("docling") and v.get("status") == "ok" and score[0] == 1 and score[1] > 0:
            break

    record["attempts"] = attempts
    record["engine"] = best_engine
    record["ocr"] = best_ocr

    if best_v is None:
        record["verification"] = {"status": "error", "measurable": False,
                                 "coverage": None, "numeric_coverage": None}
        record["ok"] = False
        record["seconds"] = round(time.time() - started, 2)
        return record

    # Recuperacion: lo que ningun motor logro traer se anexa literal.
    #
    # Se aplica siempre que falte algo, no solo cuando el documento queda por debajo del
    # umbral. El umbral sirve para clasificar la calidad de la conversion; el objetivo del
    # proyecto es que el Markdown no pierda texto. Un documento al 99,7% se considera
    # aceptable y aun asi le faltan palabras reales: darlo por bueno dejaria esas palabras
    # fuera para siempre, que es exactamente lo que la herramienta debe evitar.
    recovered_lines = 0
    if best_v.get("measurable") and best_v.get("missing_tokens"):
        block, recovered_lines = _recovery_block(reference, best_md)
        if block:
            best_md = best_md.rstrip() + block
            best_v = verify.compare(reference, best_md)
            best_v["recovered"] = True

    record["recovered_lines"] = recovered_lines
    record["verification"] = best_v
    record["ok"] = best_v.get("status") in ("ok", "no-reference")

    # Un original sin texto extraible solo puede leerse por reconocimiento optico. Si ese
    # paso fallo, el Markdown queda practicamente vacio: no hay perdida que medir porque
    # no se recupero nada, y darlo por bueno esconderia el unico caso en que la
    # herramienta no pudo leer el documento. Se marca para revision explicitamente.
    if best_v.get("status") == "no-reference":
        contenido = len(verify.tokenize(best_md))
        record["ok"] = False
        if contenido < MIN_TOKENS_SIN_REFERENCIA:
            record["verification"]["status"] = "sin-lectura"
            record["verification"]["note"] = (
                "el original no expone texto y el reconocimiento optico no produjo "
                "contenido: el documento requiere revision manual"
            )
        else:
            # Hay texto, pero proviene enteramente del reconocimiento optico: no existe
            # una version del original contra la cual medirlo. En planos rasterizados de
            # baja resolucion el reconocimiento acierta parte de las etiquetas y confunde
            # otras, y no hay forma automatica de distinguir cual es cual. Declararlo
            # conforme seria afirmar una fidelidad que nadie comprobo.
            record["verification"]["status"] = "solo-ocr"
            record["verification"]["note"] = (
                "el contenido proviene solo del reconocimiento optico; no hay texto "
                "original contra el cual verificarlo. Requiere revision visual."
            )

    target = output_root / job.rel_target
    target.parent.mkdir(parents=True, exist_ok=True)
    if job.is_chapter:
        # Un capitulo se lee suelto, asi que lleva su propio titulo y declara de donde sale.
        encabezado = (
            f"# {job.chapter_title}\n\n"
            f"> Capitulo {job.chapter_index} de *{job.rel_source.stem}* — "
            f"paginas {job.page_range[0]} a {job.page_range[1]} del original.  \n"
            f"> Documento completo: `{to_pseudopath(job.parent_target)}`\n\n"
        )
    else:
        encabezado = f"# {job.rel_source.stem}\n\n"
    body = _front_matter(job, record) + "\n\n" + encabezado + best_md.strip() + "\n"
    if compactar:
        # Se retira el andamiaje del Markdown, no el contenido: la propia funcion comprueba
        # que no falte ninguna palabra y, si faltara, devuelve el texto sin tocar.
        body, aplicada = compactar_md.compactar(body)
        record["compactado"] = aplicada
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

    if temporal is not None:
        # El recorte es material de trabajo: se descarta una vez convertido el capitulo.
        try:
            temporal.unlink(missing_ok=True)
        except Exception:
            pass

    record["seconds"] = round(time.time() - started, 2)
    return record


def convert_one_safe(args) -> dict:
    """Envoltorio para el pool de procesos: un archivo roto no debe caer el lote."""
    job, output_root, use_docling, save_lossless, *resto = args
    compactar = resto[0] if resto else True
    # Aviso de inicio: permite mostrar en pantalla que se esta trabajando ahora mismo,
    # en vez de dejar la interfaz en silencio durante un documento largo.
    etiqueta = job.rel_source.name
    if job.is_chapter:
        etiqueta += f"  [cap. {job.chapter_index}: {job.chapter_title[:40]}]"
    print(f">>> {etiqueta}", flush=True)
    try:
        return convert_one(job, output_root, use_docling, save_lossless, compactar)
    except Exception:  # noqa: BLE001
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
            "errors": [traceback.format_exc(limit=3)],
        }
