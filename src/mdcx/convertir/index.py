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

"""Generacion del indice global de la carpeta convertida.

Se emiten dos artefactos complementarios:
  - INDICE.md      : para leer. Arbol de carpetas, tabla por documento, estado de fidelidad.
  - _manifest.json : para automatizar. Mismos datos, estructurados.

Ambos identifican cada documento por pseudopath (`@/carpeta/archivo.md`), nunca por ruta
absoluta, de modo que el indice sigue siendo correcto si la carpeta se mueve de equipo,
de unidad o de nube. Los enlaces del INDICE.md son relativos, asi que funcionan al abrirlo
con cualquier visor de Markdown desde la propia carpeta.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .paths import to_pseudopath

# El indice se nombra con un cero delante para que quede primero al ordenar la carpeta:
# es lo que hay que abrir para entender el resto.
INDEX_FILENAME = "00_INDICE.md"

_STATUS_LABEL = {
    "ok": "OK",
    "warn": "REVISAR",
    "fail": "INCOMPLETO",
    "no-reference": "SIN TEXTO",
    "sin-lectura": "SIN LECTURA",
    "solo-ocr": "REVISION VISUAL",
    "error": "ERROR",
}


def _rel_link(from_dir: str, target_pseudo: str) -> str:
    """Enlace relativo desde la raiz de Output hacia el .md, apto para visores Markdown."""
    rel = target_pseudo[2:] if target_pseudo.startswith("@/") else target_pseudo
    return quote(rel)


def _fmt_pct(value) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:.2f}%"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def build_manifest(records: list[dict], input_root: Path, skipped: list[Path],
                   elapsed: float) -> dict:
    principales = [r for r in records if not r.get("es_capitulo")]
    capitulos = [r for r in records if r.get("es_capitulo")]
    ok = sum(1 for r in principales if r.get("ok"))
    # El registro de un documento dividido es un agregado de sus capitulos: sus tokens ya
    # estan contados en ellos. Incluirlo aqui contaria dos veces el mismo texto e inflaria
    # tanto lo verificado como lo faltante.
    measurable = [r for r in records
                  if (r.get("verification") or {}).get("measurable")
                  and not r.get("es_indice_de_documento")]
    coverages = [r["verification"]["coverage"] for r in measurable
                 if r["verification"].get("coverage") is not None]
    total_ref = sum(r["verification"].get("ref_tokens", 0) for r in measurable)
    total_missing = sum(r["verification"].get("missing_tokens", 0) for r in measurable)

    return {
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "esquema": "pdftomd/1",
        "convencion_pseudopath": (
            "'@/' denota la raiz de esta carpeta de salida. Resolver contra la ubicacion "
            "actual de la carpeta; nunca contiene rutas absolutas ni letras de unidad."
        ),
        "resumen": {
            "documentos": len(principales),
            "capitulos": len(capitulos),
            "convertidos_ok": ok,
            "con_observaciones": len(principales) - ok,
            "omitidos_formato_no_soportado": len(skipped),
            "cobertura_global_tokens": (
                round((total_ref - total_missing) / total_ref, 5) if total_ref else None
            ),
            "cobertura_promedio": round(sum(coverages) / len(coverages), 5) if coverages else None,
            "tokens_referencia": total_ref,
            "tokens_no_recuperados": total_missing,
            "segundos": round(elapsed, 1),
        },
        "omitidos": [p.name for p in skipped],
        "documentos": records,
    }


def write_index(records: list[dict], output_root: Path, manifest: dict) -> Path:
    # Los capitulos no se listan sueltos en el indice general: aparecen dentro del indice
    # de su propio documento, que si figura aqui como una unica entrada.
    principales = [r for r in records if not r.get("es_capitulo")]
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for r in principales:
        by_folder[r.get("folder", ".")].append(r)

    res = manifest["resumen"]
    lines: list[str] = [
        "# Indice general de documentos convertidos",
        "",
        f"Generado: {manifest['generado_utc']}  ",
        f"Documentos: **{res['documentos']}**  |  Conformes: **{res['convertidos_ok']}**  "
        f"|  Con observaciones: **{res['con_observaciones']}**  ",
        f"Cobertura global de texto: **{_fmt_pct(res['cobertura_global_tokens'])}** "
        f"({res['tokens_referencia']:,} tokens de referencia, "
        f"{res['tokens_no_recuperados']:,} no recuperados)".replace(",", "."),
        "",
        "## Como leer este indice",
        "",
        "Cada documento se identifica por **pseudopath**: una ruta portable que empieza con "
        "`@/` y que se resuelve contra la carpeta que contiene este indice, este donde este "
        "(disco local, red o nube). No hay rutas absolutas en ningun artefacto de salida.",
        "",
        "| Campo | Significado |",
        "|---|---|",
        "| `@/...md` | Documento convertido, relativo a esta carpeta |",
        "| Fidelidad | Porcentaje del texto del original presente en el Markdown |",
        "| Estado | `OK` conforme; `REVISAR` diferencias menores; `INCOMPLETO` falta contenido; "
        "`REVISION VISUAL` el texto proviene solo de reconocimiento optico y no es verificable; "
        "`SIN LECTURA` el original no expone texto y el reconocimiento no produjo nada |",
        "| Motor | Herramienta que produjo la conversion |",
        "",
        "Los documentos marcados `REVISAR` o `INCOMPLETO` incluyen al final un "
        "**Anexo de recuperacion** con el texto que el motor estructurado no traslado, "
        "copiado literal del original.",
        "",
        "---",
        "",
        "## Documentos por carpeta",
        "",
    ]

    for folder in sorted(by_folder, key=lambda f: (f != ".", f.lower())):
        rows = sorted(by_folder[folder], key=lambda r: r["source_name"].lower())
        title = "(raiz)" if folder == "." else folder
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Documento | Origen | Formato | Tam. | Fidelidad | Estado | Motor |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            v = r.get("verification") or {}
            status = _STATUS_LABEL.get(v.get("status"), v.get("status") or "?")
            link = _rel_link(folder, r["markdown_pseudopath"])
            name = r["markdown_pseudopath"].rsplit("/", 1)[-1]
            name = name.replace("|", r"\|")
            lines.append(
                f"| [{name}]({link}) | `{r['source_name']}` | {r['format']} | "
                f"{_fmt_size(r['bytes'])} | {_fmt_pct(v.get('coverage'))} | {status} | "
                f"{r.get('engine', '?')} |"
            )
        lines.append("")

    lines += ["---", "", "## Correspondencia origen -> Markdown (pseudopaths)", "",
              "| Pseudopath del original | Pseudopath del Markdown |", "|---|---|"]
    for r in sorted(principales, key=lambda r: r["source_pseudopath"].lower()):
        destino = r["markdown_pseudopath"]
        if r.get("es_indice_de_documento"):
            destino += f"  (+ {r.get('capitulos', 0)} capitulos)"
        lines.append(f"| `{r['source_pseudopath']}` | `{destino}` |")

    problems = [r for r in records if not r.get("ok")]
    if problems:
        lines += ["", "---", "", "## Documentos que requieren atencion", "",
                  "| Documento | Estado | Fidelidad | Detalle |", "|---|---|---|---|"]
        for r in sorted(problems, key=lambda r: (r.get("verification") or {}).get("coverage") or 0):
            v = r.get("verification") or {}
            detail = []
            if r.get("recovered_lines"):
                detail.append(f"{r['recovered_lines']} lineas en anexo de recuperacion")
            if v.get("missing_sample"):
                detail.append("faltan p.ej.: " + ", ".join(v["missing_sample"][:6]))
            if r.get("errors"):
                detail.append(str(r["errors"][0])[:120])
            lines.append(
                f"| `{r['markdown_pseudopath']}` | {_STATUS_LABEL.get(v.get('status'), '?')} | "
                f"{_fmt_pct(v.get('coverage'))} | {'; '.join(detail) or '-'} |"
            )

    if manifest.get("omitidos"):
        lines += ["", "---", "", "## Archivos no convertidos (formato no soportado)", ""]
        for name in manifest["omitidos"]:
            lines.append(f"- `{name}`")

    lines += ["", "---", "",
              "Datos estructurados equivalentes: `@/_manifest.json`.",
              "Respaldo sin perdida por documento (estructura completa): `@/_lossless/`.", ""]

    path = output_root / INDEX_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    (output_root / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


def write_document_index(info: dict, capitulos: list[dict], output_root: Path) -> dict:
    """Escribe el .md que agrupa los capitulos de un documento dividido.

    El documento original sigue teniendo una unica entrada en el espejo de la carpeta:
    ese archivo pasa a ser el indice, y los capitulos viven en una subcarpeta con el
    mismo nombre. Asi la correspondencia con la carpeta de origen se mantiene visible,
    y quien abra el indice encuentra el documento completo enlazado por partes.

    Devuelve el registro agregado del documento, con la fidelidad del conjunto: se pondera
    por tokens de cada capitulo, porque promediar porcentajes daria el mismo peso a un
    capitulo de una pagina que a uno de cuarenta.
    """
    capitulos = sorted(capitulos, key=lambda r: (r.get("chapter") or {}).get("indice", 0))
    rel_target: Path = info["rel_target"]
    rel_source: Path = info["rel_source"]
    carpeta_rel = rel_target.with_suffix("").name

    ref_total = sum((c.get("verification") or {}).get("ref_tokens", 0) for c in capitulos)
    faltan = sum((c.get("verification") or {}).get("missing_tokens", 0) for c in capitulos)
    cobertura = ((ref_total - faltan) / ref_total) if ref_total else None
    paginas = sum(
        (c.get("chapter") or {}).get("paginas", [0, 0])[1]
        - (c.get("chapter") or {}).get("paginas", [1, 0])[0] + 1
        for c in capitulos
    )
    conformes = sum(1 for c in capitulos if c.get("ok"))

    lineas = [
        "---",
        f'titulo: "{rel_source.stem}"',
        f'origen: "{rel_source.name}"',
        f'pseudopath_origen: "{to_pseudopath(rel_source)}"',
        f'pseudopath_markdown: "{to_pseudopath(rel_target)}"',
        "tipo: indice_de_documento",
        f"capitulos: {len(capitulos)}",
        f"paginas: {paginas}",
        f"fidelidad: {round(cobertura, 5) if cobertura is not None else 'no-medible'}",
        f"division: {'indice del documento' if info.get('desde_indice_pdf') else 'tramos de paginas'}",
        "---",
        "",
        f"# {rel_source.stem}",
        "",
        f"Documento de **{paginas} paginas**, dividido en **{len(capitulos)} capitulos** "
        + ("segun el indice del propio documento." if info.get("desde_indice_pdf")
           else "en tramos de paginas, porque el original no trae indice."),
        "",
        f"Fidelidad del conjunto: **{_fmt_pct(cobertura)}** "
        f"({conformes} de {len(capitulos)} capitulos conformes).",
        "",
        "## Capitulos",
        "",
        "| # | Capitulo | Paginas | Fidelidad | Estado |",
        "|---|---|---|---|---|",
    ]

    for c in capitulos:
        cap = c.get("chapter") or {}
        v = c.get("verification") or {}
        nombre = c["markdown_pseudopath"].rsplit("/", 1)[-1]
        # Enlace relativo desde el indice hacia la subcarpeta de capitulos.
        destino = quote(f"{carpeta_rel}/{nombre}")
        pag = cap.get("paginas") or [0, 0]
        # El titulo puede contener una barra vertical ("3. Interfaces | Integration"),
        # que partiria la fila en columnas de mas: hay que escaparla.
        etiqueta = str(cap.get("titulo") or nombre).replace("|", r"\|")
        lineas.append(
            f"| {cap.get('indice', '?')} | [{etiqueta}]({destino}) | "
            f"{pag[0]}-{pag[1]} | {_fmt_pct(v.get('coverage'))} | "
            f"{_STATUS_LABEL.get(v.get('status'), '?')} |"
        )

    lineas += [
        "",
        "---",
        "",
        f"Original: `{to_pseudopath(rel_source)}`  ",
        f"Capitulos en: `{to_pseudopath(rel_target.with_suffix(''))}/`",
        "",
    ]

    destino = output_root / rel_target
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(lineas), encoding="utf-8")

    estado = "ok" if conformes == len(capitulos) else "warn"
    return {
        "source_pseudopath": to_pseudopath(rel_source),
        "markdown_pseudopath": to_pseudopath(rel_target),
        "source_name": rel_source.name,
        "folder": rel_source.parent.as_posix() or ".",
        "format": rel_source.suffix.lower().lstrip("."),
        "bytes": capitulos[0].get("bytes", 0) if capitulos else 0,
        "engine": "division en capitulos",
        "pages": paginas,
        "es_indice_de_documento": True,
        "capitulos": len(capitulos),
        "capitulos_pseudopaths": [c["markdown_pseudopath"] for c in capitulos],
        # Detalle de los capitulos que no pasaron: sin esto, el documento aparece con una
        # fidelidad alta y marcado para revision, sin indicar donde hay que mirar.
        "capitulos_con_observacion": [
            {
                "indice": (c.get("chapter") or {}).get("indice"),
                "titulo": (c.get("chapter") or {}).get("titulo"),
                "paginas": (c.get("chapter") or {}).get("paginas"),
                "estado": (c.get("verification") or {}).get("status"),
                "cobertura": (c.get("verification") or {}).get("coverage"),
                "pseudopath": c["markdown_pseudopath"],
            }
            for c in capitulos if not c.get("ok")
        ],
        "ok": conformes == len(capitulos),
        "verification": {
            "measurable": ref_total > 0,
            "status": estado if ref_total else "no-reference",
            "coverage": round(cobertura, 5) if cobertura is not None else None,
            "numeric_coverage": None,
            "ref_tokens": ref_total,
            "missing_tokens": faltan,
            "missing_sample": [],
        },
    }


RESULTS_FILENAME = "00_RESULTADOS.txt"


def write_results_txt(records: list[dict], output_root: Path, manifest: dict,
                      elapsed: float, input_root: Path) -> Path:
    """Informe en texto plano: que documentos salieron bien y cuales requieren atencion.

    Se emite aparte del indice en Markdown porque cumple otra funcion: el indice sirve para
    navegar el contenido, este sirve para cerrar el trabajo. Quien ejecuta la conversion
    necesita una lista corta de que reviso el proceso y que quedo pendiente, legible en
    cualquier editor y facil de adjuntar en un correo.
    """
    principales = [r for r in records if not r.get("es_capitulo")]
    exito = [r for r in principales if r.get("ok")]
    fallidos = [r for r in principales if not r.get("ok")]
    res = manifest["resumen"]

    def _pct(v):
        return f"{v * 100:6.2f}%" if v is not None else "   n/d"

    L: list[str] = []
    L.append("=" * 78)
    L.append("RESULTADO DE LA CONVERSION A MARKDOWN")
    L.append("=" * 78)
    L.append(f"Fecha (UTC)      : {manifest['generado_utc']}")
    L.append(f"Carpeta de origen: {input_root}")
    L.append(f"Carpeta de salida: {output_root}")
    ejec = manifest.get("ejecucion") or {}
    if ejec:
        L.append(f"Ejecucion        : modo {ejec.get('modo')} | inferencia {ejec.get('dispositivo')} "
                 f"| procesos GPU {ejec.get('procesos_gpu')} + CPU {ejec.get('procesos_cpu')}")
    L.append(f"Duracion         : {elapsed / 60:.1f} minutos ({elapsed:.0f} s)")
    L.append("")
    L.append("-" * 78)
    L.append("RESUMEN")
    L.append("-" * 78)
    L.append(f"Documentos procesados : {res['documentos']}")
    if res.get("capitulos"):
        L.append(f"  (divididos en)      : {res['capitulos']} capitulos")
    L.append(f"Convertidos con exito : {len(exito)}")
    L.append(f"Requieren atencion    : {len(fallidos)}")
    L.append(f"Omitidos por formato  : {res['omitidos_formato_no_soportado']}")
    cg = res["cobertura_global_tokens"]
    L.append(f"Fidelidad global      : {_pct(cg)}  "
             f"({res['tokens_no_recuperados']} de {res['tokens_referencia']} tokens no recuperados)")
    L.append("")

    if fallidos:
        L.append("-" * 78)
        L.append(f"REQUIEREN ATENCION ({len(fallidos)})")
        L.append("-" * 78)
        for r in sorted(fallidos, key=lambda x: (x.get("verification") or {}).get("coverage") or 0):
            v = r.get("verification") or {}
            L.append(f"[{_STATUS_LABEL.get(v.get('status'), '?'):<10}] {_pct(v.get('coverage'))}  "
                     f"{r['source_name']}")
            L.append(f"             origen : {r['source_pseudopath']}")
            L.append(f"             salida : {r['markdown_pseudopath']}")
            if v.get("status") == "no-reference":
                L.append("             motivo : el original no expone texto (plano rasterizado); "
                         "no es medible por tokens")
            elif v.get("status") == "sin-lectura":
                L.append("             motivo : el original no expone texto y el reconocimiento "
                         "optico no produjo contenido; requiere revision manual")
            elif v.get("status") == "solo-ocr":
                L.append("             motivo : el texto proviene solo del reconocimiento optico "
                         "(plano rasterizado); no hay original contra el cual verificarlo")
            if r.get("recovered_lines"):
                L.append(f"             nota   : {r['recovered_lines']} lineas anexadas por recuperacion")
            for cap in r.get("capitulos_con_observacion") or []:
                pag = cap.get("paginas") or ["?", "?"]
                cob = cap.get("cobertura")
                L.append(f"             capitulo {cap.get('indice')}: "
                         f"{_STATUS_LABEL.get(cap.get('estado'), '?')} "
                         f"{_pct(cob) if cob is not None else '   n/d'} "
                         f"(paginas {pag[0]}-{pag[1]}) {str(cap.get('titulo') or '')[:50]}")
            if r.get("errors"):
                L.append(f"             error  : {str(r['errors'][0])[:160]}")
            L.append("")
    else:
        L.append("No hay documentos con observaciones: todos superaron la verificacion.")
        L.append("")

    L.append("-" * 78)
    L.append(f"CONVERTIDOS CON EXITO ({len(exito)})")
    L.append("-" * 78)
    for r in sorted(exito, key=lambda x: x["source_pseudopath"].lower()):
        v = r.get("verification") or {}
        extra = f" [{r.get('capitulos')} capitulos]" if r.get("es_indice_de_documento") else ""
        L.append(f"{_pct(v.get('coverage'))}  {r.get('engine', '?'):<22} {r['source_name']}{extra}")

    if manifest.get("omitidos"):
        L.append("")
        L.append("-" * 78)
        L.append(f"NO CONVERTIDOS: FORMATO NO SOPORTADO ({len(manifest['omitidos'])})")
        L.append("-" * 78)
        for n in manifest["omitidos"]:
            L.append(f"  {n}")

    L.append("")
    L.append("=" * 78)
    L.append(f"Indice navegable del contenido: {INDEX_FILENAME}")
    L.append("Detalle estructurado por documento: _manifest.json")
    L.append("=" * 78)
    L.append("")

    destino = output_root / RESULTS_FILENAME
    destino.write_text("\n".join(L), encoding="utf-8")
    return destino
