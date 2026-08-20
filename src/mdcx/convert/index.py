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

"""Global index and manifest of a converted collection.

Every document is identified by a pseudopath: a portable reference beginning
with @/ that resolves against the folder containing the index, so the output
remains valid wherever it is stored.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .paths import to_pseudopath

# The index is named with a leading zero so it sorts first in the folder.:

INDEX_FILENAME = "00_INDEX.md"

_STATUS_LABEL = {
    "ok": "OK",
    "warn": "REVISAR",
    "fail": "INCOMPLETO",
    "no-reference": "NO TEXT",
    "sin-lectura": "UNREADABLE",
    "only-ocr": "REVISION VISUAL",
    "error": "ERROR",
}

def _rel_link(from_dir: str, target_pseudo: str) -> str:
    """Relative link from the output root to the .md file, valid for Markdown viewers."""
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
    chapters = [r for r in records if r.get("es_capitulo")]
    ok = sum(1 for r in principales if r.get("ok"))
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
            "'@/' denotes the root of this output folder. Resolve against the current "
            "location of the folder; it never contains absolute paths or drive letters."
        ),
        "resumen": {
            "documents": len(principales),
            "chapters": len(chapters),
            "converted_ok": ok,
            "with_findings": len(principales) - ok,
            "skipped_archive_no_soportado": len(skipped),
            "global_token_coverage": (
                round((total_ref - total_missing) / total_ref, 5) if total_ref else None
            ),
            "mean_coverage": round(sum(coverages) / len(coverages), 5) if coverages else None,
            "reference_tokens": total_ref,
            "tokens_not_recovered": total_missing,
            "seconds": round(elapsed, 1),
        },
        "skipped": [p.name for p in skipped],
        "documents": records,
    }

def write_index(records: list[dict], output_root: Path, manifest: dict) -> Path:
    # Chapters are not listed separately in the global index: they appear inside
    # their own document, which appears here as a single entry.
    principales = [r for r in records if not r.get("es_capitulo")]
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for r in principales:
        by_folder[r.get("folder", ".")].append(r)

    res = manifest["resumen"]
    lines: list[str] = [
        "# Index of converted documents",
        "",
        f"Generated: {manifest['generado_utc']}  ",
        f"Documents: **{res['documents']}**  |  Conforming: **{res['converted_ok']}**  "
        f"|  With findings: **{res['with_findings']}**  ",
        f"Global text coverage: **{_fmt_pct(res['global_token_coverage'])}** "
        f"({res['reference_tokens']:,} reference tokens, "
        f"{res['tokens_not_recovered']:,} not recovered)".replace(",", "."),
        "",
        "## How to read this index",
        "",
        "Each document is identified by a **pseudopath**: a portable reference beginning "
        "with `@/`, resolved against the folder holding this index, wherever it is "
        "(local disk, network share or cloud). No output artefact contains absolute paths.",
        "",
        "| Campo | Significado |",
        "|---|---|",
        "| `@/...md` | Converted document, relative to this folder |",
        "| Fidelity | Percentage of the original text present in the Markdown |",
        "| Status | `OK` conforme; `REVISAR` diferencias menores; `INCOMPLETO` falta contenido; "
        "`VISUAL REVIEW` the text comes from optical recognition only and is not verifiable; "
        "`UNREADABLE` the original exposes no text and recognition produced nothing |",
        "| Engine | Tool that produced the conversion |",
        "",
        "Documents marked `REVIEW` or `INCOMPLETE` end with a "
        "**recovery section** holding the text the structured engine did not carry over, "
        "copied verbatim from the original.",
        "",
        "---",
        "",
        "## Documentos por folder",
        "",
    ]

    for folder in sorted(by_folder, key=lambda f: (f != ".", f.lower())):
        rows = sorted(by_folder[folder], key=lambda r: r["source_name"].lower())
        title = "(raiz)" if folder == "." else folder
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Documento | Source | Formato | Tam. | Fidelity | Status | Engine |")
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

    lines += ["---", "", "## Correspondencia source -> Markdown (pseudopaths)", "",
              "| Source pseudopath | Markdown pseudopath |", "|---|---|"]
    for r in sorted(principales, key=lambda r: r["source_pseudopath"].lower()):
        target = r["markdown_pseudopath"]
        if r.get("es_indice_de_documento"):
            target += f"  (+ {r.get('chapters', 0)} chapters)"
        lines.append(f"| `{r['source_pseudopath']}` | `{target}` |")

    problems = [r for r in records if not r.get("ok")]
    if problems:
        lines += ["", "---", "", "## Documents requiring attention", "",
                  "| Documento | Status | Fidelity | Detalle |", "|---|---|---|---|"]
        for r in sorted(problems, key=lambda r: (r.get("verification") or {}).get("coverage") or 0):
            v = r.get("verification") or {}
            detail = []
            if r.get("recovered_lines"):
                detail.append(f"{r['recovered_lines']} lines en anexo de recuperacion")
            if v.get("missing_sample"):
                detail.append("faltan p.ej.: " + ", ".join(v["missing_sample"][:6]))
            if r.get("errors"):
                detail.append(str(r["errors"][0])[:120])
            lines.append(
                f"| `{r['markdown_pseudopath']}` | {_STATUS_LABEL.get(v.get('status'), '?')} | "
                f"{_fmt_pct(v.get('coverage'))} | {'; '.join(detail) or '-'} |"
            )

    if manifest.get("skipped"):
        lines += ["", "---", "", "## Archivos no convertidos (archive no soportado)", ""]
        for name in manifest["skipped"]:
            lines.append(f"- `{name}`")

    lines += ["", "---", "",
              "Datos estructurados equivalentes: `@/_manifest.json`.",
              "Lossless per-document backup (full structure): `@/_lossless/`.", ""]

    path = output_root / INDEX_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    (output_root / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path

def write_document_index(info: dict, chapters: list[dict], output_root: Path) -> dict:
    """Write the .md that groups the chapters of a split document.

    The original document keeps a single entry in the folder mirror: that file
    becomes the index, and the chapters live in a subfolder of the same name, so
    the correspondence with the source folder remains visible and whoever opens
    the index finds the whole document linked in parts.

    Returns the aggregated record, with overall fidelity weighted by the tokens of
    each chapter, since averaging percentages would give a one-page chapter the
    same weight as a forty-page one.
    """
    chapters = sorted(chapters, key=lambda r: (r.get("chapter") or {}).get("outline", 0))
    rel_target: Path = info["rel_target"]
    rel_source: Path = info["rel_source"]
    carpeta_rel = rel_target.with_suffix("").name

    ref_total = sum((c.get("verification") or {}).get("ref_tokens", 0) for c in chapters)
    faltan = sum((c.get("verification") or {}).get("missing_tokens", 0) for c in chapters)
    coverage = ((ref_total - faltan) / ref_total) if ref_total else None
    pages = sum(
        (c.get("chapter") or {}).get("pages", [0, 0])[1]
        - (c.get("chapter") or {}).get("pages", [1, 0])[0] + 1
        for c in chapters
    )
    conforming = sum(1 for c in chapters if c.get("ok"))

    lines = [
        "---",
        f'title: "{rel_source.stem}"',
        f'source: "{rel_source.name}"',
        f'source_pseudopath: "{to_pseudopath(rel_source)}"',
        f'markdown_pseudopath: "{to_pseudopath(rel_target)}"',
        "tipo: indice_de_documento",
        f"chapters: {len(chapters)}",
        f"pages: {pages}",
        f"fidelity: {round(coverage, 5) if coverage is not None else 'no-medible'}",
        f"split: {'document outline' if info.get('from_pdf_outline') else 'page ranges'}",
        "---",
        "",
        f"# {rel_source.stem}",
        "",
        f"Documento de **{pages} pages**, dividido en **{len(chapters)} chapters** "
        + ("following the document's own outline." if info.get("from_pdf_outline")
           else "in page ranges, because the original has no outline."),
        "",
        f"Overall fidelity: **{_fmt_pct(coverage)}** "
        f"({conforming} of {len(chapters)} chapters conforming).",
        "",
        "## Capitulos",
        "",
        "| # | Capitulo | Paginas | Fidelity | Status |",
        "|---|---|---|---|---|",
    ]

    for c in chapters:
        cap = c.get("chapter") or {}
        v = c.get("verification") or {}
        name = c["markdown_pseudopath"].rsplit("/", 1)[-1]
        # Relative link from the index to the chapter subfolder.
        target = quote(f"{carpeta_rel}/{name}")
        pag = cap.get("pages") or [0, 0]
        # A title may contain a vertical bar, which would split the table row into
        # extra columns, so it is escaped.
        etiqueta = str(cap.get("title") or name).replace("|", r"\|")
        lines.append(
            f"| {cap.get('outline', '?')} | [{etiqueta}]({target}) | "
            f"{pag[0]}-{pag[1]} | {_fmt_pct(v.get('coverage'))} | "
            f"{_STATUS_LABEL.get(v.get('status'), '?')} |"
        )

    lines += [
        "",
        "---",
        "",
        f"Original: `{to_pseudopath(rel_source)}`  ",
        f"Capitulos en: `{to_pseudopath(rel_target.with_suffix(''))}/`",
        "",
    ]

    target = output_root / rel_target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")

    estado = "ok" if conforming == len(chapters) else "warn"
    return {
        "source_pseudopath": to_pseudopath(rel_source),
        "markdown_pseudopath": to_pseudopath(rel_target),
        "source_name": rel_source.name,
        "folder": rel_source.parent.as_posix() or ".",
        "format": rel_source.suffix.lower().lstrip("."),
        "bytes": chapters[0].get("bytes", 0) if chapters else 0,
        "engine": "division en chapters",
        "pages": pages,
        "es_indice_de_documento": True,
        "chapters": len(chapters),
        "chapters_pseudopaths": [c["markdown_pseudopath"] for c in chapters],
        "chapters_con_observacion": [
            {
                "outline": (c.get("chapter") or {}).get("outline"),
                "title": (c.get("chapter") or {}).get("title"),
                "pages": (c.get("chapter") or {}).get("pages"),
                "estado": (c.get("verification") or {}).get("status"),
                "coverage": (c.get("verification") or {}).get("coverage"),
                "pseudopath": c["markdown_pseudopath"],
            }
            for c in chapters if not c.get("ok")
        ],
        "ok": conforming == len(chapters),
        "verification": {
            "measurable": ref_total > 0,
            "status": estado if ref_total else "no-reference",
            "coverage": round(coverage, 5) if coverage is not None else None,
            "numeric_coverage": None,
            "ref_tokens": ref_total,
            "missing_tokens": faltan,
            "missing_sample": [],
        },
    }

RESULTS_FILENAME = "00_RESULTS.txt"

def write_results_txt(records: list[dict], output_root: Path, manifest: dict,
                      elapsed: float, input_root: Path) -> Path:
    """Plain-text report: which documents converted cleanly and which need attention."""
    principales = [r for r in records if not r.get("es_capitulo")]
    succeeded = [r for r in principales if r.get("ok")]
    fallidos = [r for r in principales if not r.get("ok")]
    res = manifest["resumen"]

    def _pct(v):
        return f"{v * 100:6.2f}%" if v is not None else "   n/d"

    L: list[str] = []
    L.append("=" * 78)
    L.append("MARKDOWN CONVERSION RESULT")
    L.append("=" * 78)
    L.append(f"Date (UTC)        : {manifest['generado_utc']}")
    L.append(f"Source folder     : {input_root}")
    L.append(f"Output folder     : {output_root}")
    ejec = manifest.get("ejecucion") or {}
    if ejec:
        L.append(f"Execution         : modo {ejec.get('modo')} | inference {ejec.get('dispositivo')} "
                 f"| GPU workers {ejec.get('procesos_gpu')} + CPU {ejec.get('procesos_cpu')}")
    L.append(f"Duration          : {elapsed / 60:.1f} minutes ({elapsed:.0f} s)")
    L.append("")
    L.append("-" * 78)
    L.append("SUMMARY")
    L.append("-" * 78)
    L.append(f"Documents processed    : {res['documents']}")
    if res.get("chapters"):
        L.append(f"  (divididos en)      : {res['chapters']} chapters")
    L.append(f"Converted successfully : {len(succeeded)}")
    L.append(f"Requiring attention    : {len(fallidos)}")
    L.append(f"Skipped by format      : {res['skipped_archive_no_soportado']}")
    cg = res["global_token_coverage"]
    L.append(f"Global fidelity        : {_pct(cg)}  "
             f"({res['tokens_not_recovered']} of {res['reference_tokens']} tokens not recovered)")

    # A crashed engine and an unnecessary one used to look identical here. On Linux
    # with CUDA every docling attempt failed inside the worker processes, the native
    # engine won by default, and the run reported full success with every table
    # flattened into a paragraph.
    con_fallo = [r for r in principales if r.get("failed_engines")]
    if con_fallo:
        motores = sorted({e for r in con_fallo for e in r["failed_engines"]})
        L.append(f"Engines that failed    : {len(con_fallo)} document(s), "
                 f"affecting {', '.join(motores)}")

    # Fidelity is measured in words, and a native engine recovers every word of a
    # table while flattening it. Reporting table rows makes a loss of structure
    # visible that coverage alone reports as perfect.
    filas = sum(r.get("table_rows") or 0 for r in principales)
    con_tablas = sum(1 for r in principales if (r.get("table_rows") or 0) > 0)
    L.append(f"Table rows preserved   : {filas} across {con_tablas} document(s)")
    L.append("")

    if fallidos:
        L.append("-" * 78)
        L.append(f"REQUIEREN ATENCION ({len(fallidos)})")
        L.append("-" * 78)
        for r in sorted(fallidos, key=lambda x: (x.get("verification") or {}).get("coverage") or 0):
            v = r.get("verification") or {}
            L.append(f"[{_STATUS_LABEL.get(v.get('status'), '?'):<10}] {_pct(v.get('coverage'))}  "
                     f"{r['source_name']}")
            L.append(f"             source : {r['source_pseudopath']}")
            L.append(f"             out : {r['markdown_pseudopath']}")
            if v.get("status") == "no-reference":
                L.append("             reason : the original exposes no text (raster drawing); "
                         "no es medible por tokens")
            elif v.get("status") == "sin-lectura":
                L.append("             reason : the original exposes no text and recognition "
                         "optico no produjo contenido; requiere revision manual")
            elif v.get("status") == "only-ocr":
                L.append("             reason : text comes from optical recognition only "
                         "(raster drawing); there is no original to verify it against")
            if r.get("recovered_lines"):
                L.append(f"             nota   : {r['recovered_lines']} lines anexadas por recuperacion")
            for cap in r.get("chapters_con_observacion") or []:
                pag = cap.get("pages") or ["?", "?"]
                cob = cap.get("coverage")
                L.append(f"             chapter {cap.get('outline')}: "
                         f"{_STATUS_LABEL.get(cap.get('estado'), '?')} "
                         f"{_pct(cob) if cob is not None else '   n/d'} "
                         f"(pages {pag[0]}-{pag[1]}) {str(cap.get('title') or '')[:50]}")
            if r.get("errors"):
                L.append(f"             error  : {str(r['errors'][0])[:160]}")
            L.append("")
    else:
        L.append("No documents with findings: all passed verification.")
        L.append("")

    L.append("-" * 78)
    L.append(f"CONVERTED SUCCESSFULLY ({len(succeeded)})")
    L.append("-" * 78)
    for r in sorted(succeeded, key=lambda x: x["source_pseudopath"].lower()):
        v = r.get("verification") or {}
        extra = f" [{r.get('chapters')} chapters]" if r.get("es_indice_de_documento") else ""
        L.append(f"{_pct(v.get('coverage'))}  {r.get('engine', '?'):<22} {r['source_name']}{extra}")

    if manifest.get("skipped"):
        L.append("")
        L.append("-" * 78)
        L.append(f"NO CONVERTIDOS: FORMATO NO SOPORTADO ({len(manifest['skipped'])})")
        L.append("-" * 78)
        for n in manifest["skipped"]:
            L.append(f"  {n}")

    L.append("")
    L.append("=" * 78)
    L.append(f"Navigable index of the content: {INDEX_FILENAME}")
    L.append("Detalle estructurado por document: _manifest.json")
    L.append("=" * 78)
    L.append("")

    target = output_root / RESULTS_FILENAME
    target.write_text("\n".join(L), encoding="utf-8")
    return target
